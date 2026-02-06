"""Measure pickle size of SequenceGroupMetadata in a real CoSpec run."""
import pickle
import copy
import time
import sys
sys.path.insert(0, '/workspace')

from array import array
from vllm.sequence import SequenceData, SequenceGroupMetadata


def make_sgm(seq_id, prompt_len=2048, output_len=50):
    prompt_tokens = array('l', list(range(prompt_len)))
    output_tokens = array('l', list(range(output_len)))
    sd = SequenceData.from_seqs(prompt_tokens, output_tokens)
    sd._num_computed_tokens = prompt_len + output_len
    return SequenceGroupMetadata(
        request_id=f"req_{seq_id}",
        is_prompt=False,
        seq_data={seq_id: sd},
        sampling_params=None,
        block_tables={seq_id: list(range(prompt_len // 16 + output_len // 16))},
        do_sample=True,
    )


batch_sizes = [10, 20, 30, 40]
for bs in batch_sizes:
    batch = [make_sgm(i) for i in range(bs)]

    # Measure full pickle
    t0 = time.monotonic()
    for _ in range(100):
        pickled = pickle.dumps(batch)
    t_full = (time.monotonic() - t0) / 100 * 1000
    print(f"Batch size {bs}: {len(pickled):,} bytes ({len(pickled)/1024:.1f} KB) "
          f"pickle_time={t_full:.2f}ms")

    # Measure stripped version (safe: keep _cached_all_token_ids)
    def strip_batch(batch):
        stripped = []
        for sgm in batch:
            sgm_copy = copy.copy(sgm)
            new_seq_data = {}
            for sid, sd in sgm.seq_data.items():
                sd_copy = copy.copy(sd)
                sd_copy._prompt_token_ids_tuple = ()
                sd_copy._new_appended_tokens = []
                new_seq_data[sid] = sd_copy
            sgm_copy.seq_data = new_seq_data
            sgm_copy.multi_modal_data = None
            sgm_copy.multi_modal_placeholders = None
            sgm_copy.mm_processor_kwargs = None
            sgm_copy.encoder_seq_data = None
            sgm_copy.cross_block_table = None
            stripped.append(sgm_copy)
        return stripped

    stripped_batch = strip_batch(batch)
    t0 = time.monotonic()
    for _ in range(100):
        stripped_pickled = pickle.dumps(stripped_batch)
    t_stripped = (time.monotonic() - t0) / 100 * 1000
    savings = 1 - len(stripped_pickled) / len(pickled)

    print(f"  Stripped: {len(stripped_pickled):,} bytes "
          f"({len(stripped_pickled)/1024:.1f} KB, {savings*100:.1f}% smaller) "
          f"pickle_time={t_stripped:.2f}ms")
    print()
