# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Optional, Union, cast, List
import torch
from tqdm import tqdm
import jinja2
from fastapi import Request
import os
import pandas as pd
import random

from vllm.cospec.shm import SharedMemory
from vllm.config import ModelConfig, VllmConfig, envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (CompletionLogProbs,
                                              CompletionRequest,
                                              CompletionResponse,
                                              CompletionResponseChoice,
                                              CompletionResponseStreamChoice,
                                              CompletionStreamResponse,
                                              ErrorResponse,
                                              RequestResponseMetadata,
                                              UsageInfo)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    clamp_prompt_logprobs)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import merge_async_iterators

logger = init_logger(__name__)


class OpenAIServingCompletionCoSpec(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        engine_client2: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)
        self.engine_client2 = engine_client2
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default completion sampling params from %s: %s",
                        source, self.default_sampling_params)
        
        self.dynamic_colocation = envs.COSPEC_DYNAMIC_COLOCATION
        self.colocation_mode = True
        self.selected_engine_idx = 0
        self.last_mode_switch_time = time.time()
        self.dwelling_time = 60  # 60 seconds for dynamic colocation mode switch cooldown
        self.performance_threshold = 0.05  # 5% performance difference threshold

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error
        if self.engine_client2.errored:
            raise self.engine_client2.dead_error

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response(
                "suffix is not currently supported")

        request_id = f"cmpl-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            # tokenizer same for bot clients
            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            request_prompts, engine_prompts = await self._preprocess_completion(
                request,
                tokenizer,
                request.prompt,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except TypeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except jinja2.TemplateError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                # Select engine for this prompt
                current_engine = self._select_engine()
                
                sampling_params: Union[SamplingParams, BeamSearchParams]
                default_max_tokens = self.max_model_len - len(
                    engine_prompt["prompt_token_ids"])
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        default_max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params)

                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=sampling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                if isinstance(sampling_params, BeamSearchParams):
                    generator = current_engine.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                    )
                else:
                    generator = current_engine.generate(
                        engine_prompt,
                        sampling_params,
                        request_id_item,
                        lora_request=lora_request,
                        prompt_adapter_request=prompt_adapter_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        model_name = self._get_model_name(request.model, lora_request)
        num_prompts = len(engine_prompts)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. Noting that best_of is only supported in V0. In addition,
        # we do not stream the results when use beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        # Streaming response
        if stream:
            return self.completion_stream_generator(
                request,
                result_generator,
                request_id,
                created_time,
                model_name,
                num_prompts=num_prompts,
                tokenizer=tokenizer,
                request_metadata=request_metadata)

        # Non-streaming response
        final_res_batch: list[Optional[RequestOutput]] = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    final_res.prompt = request_prompts[i]["prompt"]

            final_res_batch_checked = cast(list[RequestOutput],
                                           final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
                request_metadata,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        result_generator: AsyncIterator[tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_text_lens = [0] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts
        num_prompt_tokens = [0] * num_prompts

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage
            include_continuous_usage = include_usage and \
                                       stream_options.continuous_usage_stats
        else:
            include_usage, include_continuous_usage = False, False

        try:
            async for prompt_idx, res in result_generator:
                prompt_token_ids = res.prompt_token_ids
                prompt_logprobs = res.prompt_logprobs
                prompt_text = res.prompt

                # Prompt details are excluded from later streamed outputs
                if res.prompt_token_ids is not None:
                    num_prompt_tokens[prompt_idx] = len(res.prompt_token_ids)

                delta_token_ids: GenericSequence[int]
                out_logprobs: Optional[GenericSequence[Optional[dict[
                    int, Logprob]]]]

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices

                    assert request.max_tokens is not None
                    if request.echo and not has_echoed[i]:
                        assert prompt_token_ids is not None
                        assert prompt_text is not None
                        if request.max_tokens == 0:
                            # only return the prompt
                            delta_text = prompt_text
                            delta_token_ids = prompt_token_ids
                            out_logprobs = prompt_logprobs
                        else:
                            assert prompt_logprobs is not None
                            # echo the prompt and first token
                            delta_text = prompt_text + output.text
                            delta_token_ids = [
                                *prompt_token_ids, *output.token_ids
                            ]
                            out_logprobs = [
                                *prompt_logprobs,
                                *(output.logprobs or []),
                            ]
                        has_echoed[i] = True
                    else:
                        # return just the delta
                        delta_text = output.text
                        delta_token_ids = output.token_ids
                        out_logprobs = output.logprobs

                        if not delta_text and not delta_token_ids \
                            and not previous_num_tokens[i]:
                            # Chunked prefill case, don't return empty chunks
                            continue

                    if request.logprobs is not None:
                        assert out_logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_completion_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            tokenizer=tokenizer,
                            initial_text_offset=previous_text_lens[i],
                            return_as_token_id=request.
                            return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    previous_text_lens[i] += len(output.text)
                    previous_num_tokens[i] += len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                stop_reason=stop_reason,
                            )
                        ])
                    if include_continuous_usage:
                        prompt_tokens = num_prompt_tokens[prompt_idx]
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )

                    response_json = chunk.model_dump_json(exclude_unset=False)
                    yield f"data: {response_json}\n\n"

            total_prompt_tokens = sum(num_prompt_tokens)
            total_completion_tokens = sum(previous_num_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens)

            if include_usage:
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=final_usage_info,
                )
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=False, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = final_usage_info

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: list[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> CompletionResponse:
        choices: list[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0

        for final_res in final_res_batch:
            prompt_token_ids = final_res.prompt_token_ids
            assert prompt_token_ids is not None
            prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
            prompt_text = final_res.prompt

            token_ids: GenericSequence[int]
            out_logprobs: Optional[GenericSequence[Optional[dict[int,
                                                                 Logprob]]]]

            for output in final_res.outputs:
                assert request.max_tokens is not None
                if request.echo:
                    assert prompt_text is not None
                    if request.max_tokens == 0:
                        token_ids = prompt_token_ids
                        out_logprobs = prompt_logprobs
                        output_text = prompt_text
                    else:
                        token_ids = [*prompt_token_ids, *output.token_ids]

                        if request.logprobs is None:
                            out_logprobs = None
                        else:
                            assert prompt_logprobs is not None
                            assert output.logprobs is not None
                            out_logprobs = [
                                *prompt_logprobs,
                                *output.logprobs,
                            ]

                        output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    out_logprobs = output.logprobs
                    output_text = output.text

                if request.logprobs is not None:
                    assert out_logprobs is not None, "Did not output logprobs"
                    logprobs = self._create_completion_logprobs(
                        token_ids=token_ids,
                        top_logprobs=out_logprobs,
                        tokenizer=tokenizer,
                        num_output_top_logprobs=request.logprobs,
                        return_as_token_id=request.return_tokens_as_token_ids,
                    )
                else:
                    logprobs = None

                choice_data = CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    prompt_logprobs=final_res.prompt_logprobs,
                )
                choices.append(choice_data)

                num_generated_tokens += len(output.token_ids)

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        request_metadata.final_usage_info = usage

        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

    def _create_completion_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[dict[int, Logprob]]],
        num_output_top_logprobs: int,
        tokenizer: AnyTokenizer,
        initial_text_offset: int = 0,
        return_as_token_id: Optional[bool] = None,
    ) -> CompletionLogProbs:
        """Create logprobs for OpenAI Completion API."""
        out_text_offset: list[int] = []
        out_token_logprobs: list[Optional[float]] = []
        out_tokens: list[str] = []
        out_top_logprobs: list[Optional[dict[str, float]]] = []

        last_token_len = 0

        should_return_as_token_id = return_as_token_id if \
            return_as_token_id is not None else self.return_tokens_as_token_ids
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"

                out_tokens.append(token)
                out_token_logprobs.append(None)
                out_top_logprobs.append(None)
            else:
                step_token = step_top_logprobs[token_id]

                token = self._get_decoded_token(
                    step_token,
                    token_id,
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )
                token_logprob = max(step_token.logprob, -9999.0)

                out_tokens.append(token)
                out_token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                out_top_logprobs.append({
                    # Convert float("-inf") to the
                    # JSON-serializable float that OpenAI uses
                    self._get_decoded_token(top_lp[1],
                                            top_lp[0],
                                            tokenizer,
                                            return_as_token_id=should_return_as_token_id):
                    max(top_lp[1].logprob, -9999.0)
                    for i, top_lp in enumerate(step_top_logprobs.items())
                    if num_output_top_logprobs >= i
                })

            if len(out_text_offset) == 0:
                out_text_offset.append(initial_text_offset)
            else:
                out_text_offset.append(out_text_offset[-1] + last_token_len)
            last_token_len = len(token)

        return CompletionLogProbs(
            text_offset=out_text_offset,
            token_logprobs=out_token_logprobs,
            tokens=out_tokens,
            top_logprobs=out_top_logprobs,
        )
    
    async def cospec_profile(self) -> None:
        if envs.COSPEC_DYNAMIC_COLOCATION:
            await self.profile_colocation()

        if envs.COSPEC_SELECTIVE_VALIDATION:
            await self.profile_tiling()
    
    """
    Profiler for dynamic colocation. 
    """
    async def profile_colocation(self) -> None:
        loaded_cached_profile = await self.engine_client.maybe_load_cached_cospec_profile()
        if loaded_cached_profile:
            logger.info("Loaded cached profile. Skipping profiling.")
            return 

        vllm_config = await self.engine_client.get_vllm_config()
        vllm_config2 = await self.engine_client2.get_vllm_config()
        original_num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
        original_num_speculative_tokens2 = vllm_config2.speculative_config.num_speculative_tokens

        batch_sizes = range(8, vllm_config.scheduler_config.max_num_seqs // 2 + 1, 8)
        num_speculative_tokens_list = range(1, 8)

        # Set tqdm as a single progress bar
        total_iterations = len(batch_sizes) * len(num_speculative_tokens_list)

        # warmup
        await self._profile_non_colocation(8, 1, 128)
        await self._profile_colocation(8, 1, 128)

        await self.engine_client.start_cospec_profile(mode="colocation")
        await self.engine_client2.start_cospec_profile(mode="colocation")

        with tqdm(total=total_iterations, desc="Profiling...") as pbar:
            for batch_size in batch_sizes:
                for num_speculative_tokens in num_speculative_tokens_list:
                    await self._profile_non_colocation(batch_size, num_speculative_tokens, 128)
                    await self._profile_colocation(batch_size, num_speculative_tokens, 128)
                    pbar.update(1)

        await self.engine_client.stop_cospec_profile()
        await self.engine_client2.stop_cospec_profile()

        # reset num_speculative_tokens
        await self.engine_client.set_num_speculative_tokens(original_num_speculative_tokens)
        await self.engine_client2.set_num_speculative_tokens(original_num_speculative_tokens2)

    async def _profile_non_colocation(self, batch_size: int, num_speculative_tokens: int, max_tokens: int):
        await self.engine_client.set_colocation_mode(False)
        await self.engine_client2.set_colocation_mode(False)
        await self.engine_client.set_num_speculative_tokens(num_speculative_tokens)
        await self.engine_client2.set_num_speculative_tokens(num_speculative_tokens)
        await self.engine_client.set_profile_batch_size(batch_size)
        await self.engine_client2.set_profile_batch_size(batch_size)

        generators: list[AsyncGenerator[RequestOutput, None]] = []

        for i in range(batch_size):
            request_id = f"profile_{i}"
            dummy_prompt = TokensPrompt(prompt_token_ids=[1])
            sampling_params = SamplingParams(temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=max_tokens)
            generator = self.engine_client.generate(prompt=dummy_prompt, sampling_params=sampling_params, request_id=request_id)
            generators.append(generator)
        
        result_generator = merge_async_iterators(*generators)

        async for _ in result_generator:
            pass

    async def _profile_colocation(self, batch_size: int, num_speculative_tokens: int):
        await self.engine_client.set_colocation_mode(True)
        await self.engine_client2.set_colocation_mode(True)
        await self.engine_client.set_num_speculative_tokens(num_speculative_tokens)
        await self.engine_client2.set_num_speculative_tokens(num_speculative_tokens)
        await self.engine_client.set_profile_batch_size(batch_size)
        await self.engine_client2.set_profile_batch_size(batch_size)

        generators: list[AsyncGenerator[RequestOutput, None]] = []

        engine_idx = 0
        for i in range(batch_size):
            request_id = f"profile_{i}"
            dummy_prompt = TokensPrompt(prompt_token_ids=[1])
            sampling_params = SamplingParams(temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=128)
            engine_idx = (engine_idx + 1) % 2
            current_engine = self.engine_client if engine_idx == 0 else self.engine_client2
            generator = current_engine.generate(prompt=dummy_prompt, sampling_params=sampling_params, request_id=request_id)
            generators.append(generator)

        result_generator = merge_async_iterators(*generators)

        async for _ in result_generator:
            pass

    """
    Profiler for tiled selective validation.
    """
    async def profile_tiling(self):
        loaded_cached_profile = await self.engine_client.maybe_load_cached_tiling_profile()
        if loaded_cached_profile:
            logger.info("Loaded cached tiling profile. Skipping profiling.")
            loaded_cached_profile = await self.engine_client2.maybe_load_cached_tiling_profile()
            assert loaded_cached_profile, "Cached tiling profile cannot be loaded on secondary engine"
            return 
        
        vllm_config = await self.engine_client.get_vllm_config()
        original_num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

        # Fixed batch size at max_num_seqs
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        
        # Target total token counts in multiples of 8 up to 2048
        target_token_counts = list(range(8, max_num_batched_tokens + 1, 8))
        
        # Calculate prompt lengths for each target token count
        prompt_lengths_list = []
        batch_sizes = []
        for target_tokens in target_token_counts:
            if target_tokens <= max_num_seqs:
                # When target tokens <= max_num_seqs, use target_tokens as batch size
                batch_size = target_tokens
                prompt_lengths = [1] * batch_size
            else:
                # When target tokens > max_num_seqs, distribute tokens across sequences
                batch_size = max_num_seqs
                base_length = target_tokens // max_num_seqs
                remainder = target_tokens % max_num_seqs
                
                # Create list with base_length for all sequences
                prompt_lengths = [base_length] * max_num_seqs
                
                # Distribute remainder tokens across first 'remainder' sequences
                for i in range(remainder):
                    prompt_lengths[i] += 1
            
            prompt_lengths_list.append(prompt_lengths)
            batch_sizes.append(batch_size)
        
        # Set tqdm as a single progress bar
        total_iterations = len(target_token_counts)

        # warmup with minimal prompt length
        await self._profile_tiling(8, [1] * 8)  # Use smallest batch size for warmup

        await self.engine_client.start_cospec_profile(mode="tiling")

        with tqdm(total=total_iterations, desc="Profiling...") as pbar:
            for prompt_lengths, batch_size in zip(prompt_lengths_list, batch_sizes):
                await self._profile_tiling(batch_size, prompt_lengths)
                pbar.update(1)

        await self.engine_client.stop_cospec_profile()

        loaded_cached_profile = await self.engine_client2.maybe_load_cached_tiling_profile()
        assert loaded_cached_profile, "Cached tiling profile cannot be loaded on secondary engine"

        # reset num_speculative_tokens
        await self.engine_client.set_num_speculative_tokens(original_num_speculative_tokens)

    async def _profile_tiling(self, batch_size: int, prompt_lengths: List[int]): 
        await self.engine_client.set_colocation_mode(False)
        await self.engine_client2.set_colocation_mode(False)
        await self.engine_client.set_num_speculative_tokens(0)
        await self.engine_client2.set_num_speculative_tokens(0)
        await self.engine_client.set_profile_batch_size(batch_size)
        await self.engine_client2.set_profile_batch_size(batch_size)

        generators: list[AsyncGenerator[RequestOutput, None]] = []

        for _ in range(16):
            for i in range(batch_size):
                request_id = f"profile_{i}"
                # Create a prompt with the specified length for this sequence
                prompt_length = prompt_lengths[i]
                dummy_prompt = TokensPrompt(prompt_token_ids=[1] * prompt_length)
                sampling_params = SamplingParams(temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=1)
                generator = self.engine_client.generate(prompt=dummy_prompt, sampling_params=sampling_params, request_id=request_id)
                generators.append(generator)
            
            result_generator = merge_async_iterators(*generators)

            async for _ in result_generator:
                pass

    async def is_selective_validator_trained(self) -> bool:
        if envs.COSPEC_SELECTIVE_VALIDATION:
            return await self.engine_client.is_selective_validator_trained()
        else:
            return True

    def _select_engine(self) -> EngineClient:
        if self.dynamic_colocation:
            self._maybe_change_colocation_mode()
            if self.colocation_mode:
                return self._load_balance()
            else:
                return self.engine_client if self.selected_engine_idx == 0 else self.engine_client2
        else:
            return self._load_balance()

    def _load_balance(self) -> EngineClient:
        engine_client1_num_requests = self.engine_client.get_num_requests()
        engine_client2_num_requests = self.engine_client2.get_num_requests()
        return self.engine_client if engine_client1_num_requests < engine_client2_num_requests else self.engine_client2

    def _maybe_change_colocation_mode(self):
        current_time = time.time()
        elapsed = current_time - self.last_mode_switch_time
        
        # Enforce minimum dwelling time
        if elapsed < self.dwelling_time:
            return
        
        total_requests = self.engine_client.get_num_requests() + self.engine_client2.get_num_requests()
        
        ratio = self.engine_client.predict_colocation_speedup_ratio(total_requests)
        print(f"Colocation speedup ratio: {ratio}")
        
        switched = False
        if self.colocation_mode:
            if ratio < (1 - self.performance_threshold):
                print(f"Switching to non-colocation mode")
                self.colocation_mode = False
                # Select the engine with more requests
                engine1_requests = self.engine_client.get_num_requests()
                engine2_requests = self.engine_client2.get_num_requests()
                self.selected_engine_idx = 0 if engine1_requests >= engine2_requests else 1
                switched = True
            
        else: 
            if ratio > (1 + self.performance_threshold):
                print(f"Switching to colocation mode")
                self.colocation_mode = True
                switched = True

        if switched:
            self.last_mode_switch_time = current_time
            print(f"Mode switched to {'colocation' if self.colocation_mode else 'non-colocation'}")
