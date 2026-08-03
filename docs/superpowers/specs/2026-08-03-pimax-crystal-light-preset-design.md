# Pimax Crystal Light Headset Preset

## Goal

Add Pimax Crystal Light to the existing VR headset preset selector using the
same native-display semantics as the other headset entries.

## Verified Hardware Values

- Name: Pimax Crystal Light
- Resolution per eye: 2880 x 2880
- Horizontal field of view: 110 degrees
- Official source: https://eu.pimax.com/pages/crystal-light

Refresh rate is not part of the preset because output frame rate is controlled
independently by the source and encoding settings.

## Behavior

Selecting `Pimax Crystal Light` will:

- Select custom VR resolution.
- Populate custom width with `2880`.
- Populate custom height with `2880`.
- Set fisheye FOV to `110`.

It will not change stereo baseline, focal length, distortion, depth model,
frame rate, or video format.

## Implementation

- Add a `pimax-crystal-light` entry to `VR_HEADSET_PRESETS`.
- Add the preset to the headset selector and its browser-side preset map.
- Repair the existing handler so it writes to `customWidth` and `customHeight`
  instead of the nonexistent `customResolution` element.
- Keep the existing reset behavior for `None (Manual Settings)`.

## Verification

- Unit-test the canonical preset values in `VR_HEADSET_PRESETS`.
- Test or statically verify that the rendered selector and browser-side map
  contain the new preset.
- Verify selecting a preset targets the two existing custom resolution fields.
- Run the complete test suite and lint checks.

## Non-Goals

- No automatic change to stereo strength or lens distortion.
- No additional animation-specific Pimax preset.
- No attempt to match headset refresh rate during video encoding.
