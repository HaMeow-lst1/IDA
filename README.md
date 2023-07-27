# Offical pytorch code for paper "Illumination Distribution-Aware Thermal Pedestrian Detection".

## Abstract

Thermal pedestrian detection is challenging due to the significant effect of temperature variation on the illumination of images and that fine-grained illumination annotations are difficult to acquire. Existing methods have attempted to exploit coarse-grained day/night labels, which however even hampers the model performance. In this work, we introduce a novel idea of regressing conditional thermal-visible feature distribution, dubbed as ***Illumination Distribution-Aware adaptation***(IDA). The key idea is to predict the conditional visible feature distribution given a thermal image, subject to their pre-computed joint distribution. Specifically, we first estimate the thermal-visible feature joint distribution by  constructing feature co-occurrence matrices, offering a conditional probability distribution for any given thermal image. With this pairing information, we then form a conditional probability distribution regression task for model optimization. Critically, as a model agnostic strategy, this allows the visible feature knowledge to be transferred to the thermal counterpart implicitly for learning more discriminating feature representation. Experiment results show that our method outperforms the prior art methods even using extra illumination annotations. Code is available at <https://github.com/HaMeow-lst1/IDA>.

## Appendix

The "Appendix.pdf" contains more analyze for FasterRCNN and CascadeRCNN on the KAIST and FLIR-aligned dataset.
