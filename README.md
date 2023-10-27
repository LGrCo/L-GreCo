# L-GreCo: AN EFFICIENT AND GENERAL FRAMEWORK FOR LAYERWISE-ADAPTIVE GRADIENT COMPRESSION
<div style="text-align: justify;">
Data-parallel distributed training of deep neural networks (DNN) has gained very widespread adoption, but
can still experience communication bottlenecks due to gradient transmission. To address this issue, entire
families of lossy gradient compression mechanisms have been developed, including quantization, sparsification,
and low-rank approximation, some of which are seeing significant practical adoption. Despite this progress,
almost all known compression schemes apply compression uniformly across DNN layers, although layers are
heterogeneous in terms of parameter count and their impact on model accuracy. In this work, we provide
a general framework for adapting the degree of compression across the model’s layers dynamically during
training, significantly improving the overall compression without sacrificing accuracy. Our framework, called
L-GreCo, is based on an efficient adaptive algorithm, which automatically picks the optimal compression
parameters for model layers guaranteeing the best compression ratio while respecting a theoretically-justified
error constraint. Our extensive experimental study over image classification and language modeling tasks shows
that L-GreCo is effective across all three compression families, and achieves up to 2.5× training speedup and
up to 5× compression improvement over efficient implementations of standard approaches while recovering
full accuracy. Moreover, we show that L-GreCo is complementary to existing adaptive algorithms improving
their compression ratio by 50% and practical throughput by 66%. <br />
  
  
For ResNet50 and Transformer-XL experiments we used https://github.com/NVIDIA/DeepLearningExamples, for Transformer-LM experiments we used https://github.com/facebookresearch/fairseq, and for ResNet18 experiment we used the ResNet18 model provided https://github.com/epfml/powersgd. <br />

  
You can find all hooks in a folder named 'hooks', and also you can use 'lgreco.sh' file in each experiment in order to run the task with lgreco. ResNet18 is a natural entery point.
</div>
