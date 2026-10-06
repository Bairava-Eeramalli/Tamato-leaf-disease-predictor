# Image validation manual test checklist

Run the app with `streamlit run app/app.py` and click **Predict Disease** for
each upload. The current project does not contain `models/leaf_binary.keras`,
so MobileNetV2 is diagnostic-only and the app conservatively returns
“uncertain” for every image instead of calling the disease CNN.

- [ ] Clear tomato leaf: inspect the top-five ImageNet labels and confirm that
      the app asks for a clearer image unless a binary validator is installed.
- [ ] Dog: confirm no disease result is shown and the CNN is not called.
- [ ] Person: confirm no disease result is shown and the CNN is not called.
- [ ] Car: confirm no disease result is shown and the CNN is not called.
- [ ] Blurry image: confirm no disease result is shown.

## Dedicated validator setup

Train a separate binary classifier on representative tomato-leaf positives
and unrelated negatives (dogs, people, cars, backgrounds, and blurry images).
Use RGB `224x224` inputs with the same normalization used during training and
export it as `models/leaf_binary.keras`. Its single output must be a sigmoid
probability where higher means “leaf”. Calibrate the operating threshold on a
held-out validation set rather than choosing it to fit a few manual examples.

The app accepts probabilities strictly above or below `0.5`; exactly `0.5` is
uncertain. Change that operating point only together with measured validation
metrics. MobileNetV2/ImageNet remains a diagnostic model and is not a
replacement for this binary classifier. The five-class CNN’s softmax
confidence is never used as leaf evidence.
