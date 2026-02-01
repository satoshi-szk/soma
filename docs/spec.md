# SOMA Spec

Sonic Observation, Musical Abstraction

## Overview

A GUI application that extracts partials specified by humans on a spectrogram and exports them in several formats such as MPE MIDI.

## Example input audio

- Environmental sounds with mixed noise and pitch (e.g. rain hitting a gutter)
- Sources with pitch or harmony such as instruments or vocals
- Modes and

## Example workflow

- Wavelet transform -> spectrogram display
- The user traces desired partials on the spectrogram with a pencil tool (mouse drag)
- The set of ridge frequencies/amplitudes closest to the traced path is grouped and recorded as a partial (ridge snapping)
- When the user chooses export, partials are written as MPE SMF, etc. One partial corresponds to one note.

## Example export formats

- SMF (MPE / Multi-Track MIDI / Monophonic MIDI)
- Audio files (for modular synthesizer CV)

## MVP feature list

- Spectrogram display
- Specify partials with a pencil tool
- Edit partials: delete, merge, extend, crop
- Undo / Redo for edits
- Playback of input audio files
- Playback of partials (resynthesis with sine waves)
- Mix playback of original + resynthesis (one knob)
- Per-partial mute / unmute

## Features to add later

- Analysis using Vamp plugins
- Export time-series data from Vamp plugins or built-in analyzers as CV

# Technical Spec

- macOS and Windows required
- Linux if possible

## Notes

- Not a commercial app. A research/creative tool.
- Snapping
    - While drawing (dragging), do not snap; show the raw mouse path.
    - On `MouseUp`, send the path to the backend, detect ridges (local maxima) via JIT analysis, then snap.
    - **Important:** Snapping must be done only with CWT-based JIT analysis, and must never use STFT (STFT is GUI preview only).
- Spectrogram (preview) generation
    - **Policy:** First show a low-quality STFT preview, and only when the display window length `time_end - time_start` is below a threshold, replace it in the background with a high-quality CWT preview.
    - **Comms:** The frontend sends fire-and-forget requests on init/zoom, and the backend pushes preview updates (no frontend polling).
- Undo / Redo
    - Only changes to the document (partials + analysis settings) are subject to Undo/Redo.
    - **Pure UI state** such as selection, zoom/pan, and view modes is not included in Undo/Redo.
    - History (Undo/Redo stack) is managed by the backend; the frontend sends command/Undo/Redo requests.
- Synthesis (preview)
    - Because destructive additive synthesis is used, small numerical errors may remain after repeated Undo/Redo, but this is acceptable for preview.
- MPE supports up to 15 voices. If 16 or more partials sound at the same time, split into multiple SMF files.
    - Export should be auto-split with numbered filenames using the same base name (e.g. `project_01.mid`, `project_02.mid`).
- Multi-Track MIDI uses 1 track = 1 voice, and all tracks output on the same channel.
- Monophonic MIDI uses 1 track = 1 channel, and allows overlapping notes (expecting legato handling on the mono synth side).
- Allow choosing which MPE CC to assign partial amplitude to at export time.
- Do not do extra things like auto-muting low amplitudes or auto-splitting partials. Respect user-drawn partials and enforce 1 partial = 1 note.
- No real-time playback. After each edit, resynthesize sine waves in the background. If the user hits play during resynthesis, wait until background processing completes. Edits during playback take effect only after stopping and playing again.
- Assume 10 to about 10,000 partials.
- Resynthesis can ignore phase. Smoothly connect partial freq/amplitude, and apply about a 5ms fade-in/out at endpoints.
