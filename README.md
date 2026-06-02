In the era of social media and online communication, hateful content is widely
propagated, which can negatively impact users’ sensitivity. However, human beings
cannot manually filter all such content due to the enormous volume of data being
generated. As a result, automated detection systems are necessary for detecting
and filtering hateful content.
The Hateful Memes Challenge, introduced by Facebook AI (1), is a multimodal
classification task. It consists of building a model capable of determining whether a
meme is hateful or not. This challenge is specifically designed such that unimodal
models are insufficient for this task, thereby requiring multimodal approaches that
jointly process textual and visual information.
In this project, we propose a multimodal model that leverages a pretrained Dis
tilBERT (7) model as a text encoder and a pretrained ResNet-18 (8) as a visual
encoder. Our architecture incorporates a bidirectional cross-attention module (6)
to integrate textual and visual representations by capturing interactions between
them. We also experiment different fusion strategies for combining image and text
embeddings, and evaluate the impact of incorporating CLIP embeddings.
Finally, we compare our approach with pretrained CLIP (3) and FLAVA (2) ar
chitectures, using only trainable classification heads. This allows us to assess
the effectiveness of our trainable intermediate-fusion strategy against FLAVA’s
pretrained intermediate-fusion and CLIP’s late-fusion of aligned embeddings, both
of which are state-of-the-art multimodal models.

We achieved a F1 score of 71% and a AUROC score of 0.78 for our DistilBRT-ResNET model, which is better than the CLIP or FLAVA pretrained models.

You will find the pdf file of the report of this project in the repository.
