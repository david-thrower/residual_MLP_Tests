?	&?"?d?P@&?"?d?P@!&?"?d?P@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?"?d?P@1????2@I??u???G@r0*	??n?0o@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice]¡?xx??!?=X{>@)]¡?xx??1?=X{>@:Preprocessing2T
Iterator::Root::ParallelMapV2"?
?l??!??o?2@)"?
?l??1??o?2@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapv?e??S??!????F@)2Ƈ?˶??1u?U??.@:Preprocessing2E
Iterator::Root?????:??!?rXp3?@@)???>r??1??_??q.@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatMۿ?Ҥ??!?s[??(0@)?-?l???1?0?J+@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip)x
?R??!???G??P@)??U?P???1??-?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?)?D/?x?!?ۚ?I@)?)?D/?x?1?ۚ?I@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?72.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????)R@Q?,???Y;@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	????2@????2@!????2@*      ??!       2      ??!       :	??u???G@??u???G@!??u???G@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????)R@y?,???Y;@?"N
2model_4/model/multi_head_attention_1/einsum/EinsumEinsum?[??t?!?[??t?"N
2model_4/model/multi_head_attention_3/einsum/EinsumEinsum?[??t?!?[????"N
2model_4/model/multi_head_attention_4/einsum/EinsumEinsum!iN_k?t?!?1?:??"N
2model_4/model/multi_head_attention_2/einsum/EinsumEinsum???^??t?!D?:=ץ??"L
0model_4/model/multi_head_attention/einsum/EinsumEinsumʼ~ؐt?!v??\ʙ?"^
Bgradient_tape/model_4/model/multi_head_attention_1/einsum_1/EinsumEinsum(?U?s?!@%????"^
Bgradient_tape/model_4/model/multi_head_attention_4/einsum_1/EinsumEinsum4q?Y
s?!ƴ?6????"\
@gradient_tape/model_4/model/multi_head_attention/einsum_1/EinsumEinsumKHy?r?!o??Y???"^
Bgradient_tape/model_4/model/multi_head_attention_3/einsum_1/EinsumEinsumcJ??r?!??;c?e??"^
Bgradient_tape/model_4/model/multi_head_attention_2/einsum_1/EinsumEinsum???/??r?!??:??¨?I???K???Qr?O;??X@YV???f???a8?e"?X@q?U?3?X@yAd~????"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?72.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 