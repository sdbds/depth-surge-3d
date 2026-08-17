# VDPP Third-Party Notice

This directory contains the minimal inference subset of VDPP, distributed
under the Apache License 2.0.

- Upstream: https://github.com/injun-baek/VDPP
- Release: v1.0
- Revision: `73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5`
- Copyright: the VDPP authors and the copyright holders named in individual
  source-file headers.

Packaging changes are limited to package markers and relative imports recorded
in `UPSTREAM.json`. The demo, assets, image-depth model, and visualization code
are not included.

The integration adapter outside this vendored directory registers an inert
`shift_head` slot for two zero-valued tensors present in the released v1.0
checkpoint but absent from the pinned public class. The vendored forward path
is unchanged and does not call that slot.
