"""
infer_flux2_steer.py with plain additive steering: the one rescale that brings every edited write back to its
original length (flux2/residual_steer.py, `_rescale`) is switched off at runtime, so each term adds its vector
and the norm grows with the strength. For the rotation-vs-additive strength strip (paper/experiment.md).
Does not modify infer_flux2_steer.py or flux2/residual_steer.py; every flag keeps its meaning.

Instruction term only (Δ_sem = sem_lambda |d_t| u_k added, no rescale), same anchor as the rotation runs:
    python infer_flux2_additive.py --use_masks --free_latent --free_context --strict_masks --ortho_alpha 0 \
        --sem_ref target --sem_lambda 2 --sem_double 0,1,2,3,4 --sem_double_beta 2 --sample_ids 16 --exp_name add_l2
"""
import infer_flux2_steer as base
import flux2.residual_steer as residual_steer


def _no_rescale(old, new):
    return new


if __name__ == "__main__":
    residual_steer._rescale = _no_rescale
    base.main()
