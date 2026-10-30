"""IndoML 2026 Datathon - Track 1 noise-event detection: the shared inference package.

Modules
  common            paths, constants, jsonl / zip / wav helpers
  p2_infer          per-frame inference of a P2-architecture model at an input shift
  refine            exact P2 decoder replay + the label-free refiner
  fuse              weighted interval fusion (1-D weighted box fusion)
  templates         noise-library builder and matched-filter (NCC) template detector
                    (used to build the training library, and by the 950232 synthetic branch)
  router            the file-name router (950232 only: the template-decode base and the SR1 step; C8 has no router)
  score             the organizer metric (event F1 + segment Dice), as published by the organizers
  build_submission  wav durations (<split>-durations.json) and the template decode of the synthetic-routed clips
                    (the base of the 950232 synthetic branch)

See the README of this track for the commands.
"""

__all__ = ['common', 'templates', 'p2_infer', 'refine', 'fuse', 'router', 'score', 'build_submission']
__version__ = '1.0.0'
