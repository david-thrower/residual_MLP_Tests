?	V??Ά?D@V??Ά?D@!V??Ά?D@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsV??Ά?D@1C?ʠ??
@I??$y0C@r0*	?Q???q@2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??h8e??!???? k>@)??h8e??1???? k>@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMape??)1??!ć??~?L@)F?-t%??1???e7@:Preprocessing2T
Iterator::Root::ParallelMapV2?ͮ{+??!K?Խܭ(@)?ͮ{+??1K?Խܭ(@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???{??!?yH?v*@)z?sѐ???1+?r"|'@:Preprocessing2E
Iterator::Root??????!??k*,7@)?qm????1?:ew?%@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipo.??'H??!??{?4S@)BȗP???1uKԨ@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??i??_??!q??~;A@);oc?#Շ?1w??;m/@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorgd??St?!d?60???)gd??St?1d?60???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?92.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIĪ???V@Qݩ"h @Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	C?ʠ??
@C?ʠ??
@!C?ʠ??
@*      ??!       2      ??!       :	??$y0C@??$y0C@!??$y0C@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qĪ???V@yݩ"h @?"C
*model/batch_normalization_12/batchnorm/mulMul???ro?!???ro?"A
%gradient_tape/model/dense_15/MatMul_1MatMul?lL??Am?!?"?NZ~?"A
%gradient_tape/model/dense_14/MatMul_1MatMulU)=??-m?!HX`??x??"@
$gradient_tape/model/dense_7/MatMul_1MatMul??-?m?!? ?b???"F
+model/batch_normalization_8/batchnorm/add_1AddV2???>9k?!?k	4D??"_
>gradient_tape/model/batch_normalization_10/moments/BroadcastToBroadcastTo? B??Di?!23???l??"Q
0gradient_tape/model/dense_16/BiasAdd/BiasAddGradBiasAddGrad??{P<@h?!ѥ?*?t??"X
?gradient_tape/model/batch_normalization_1/batchnorm/add_1/Sum_1Sum z??g?!????h??"P
/gradient_tape/model/dense_1/BiasAdd/BiasAddGradBiasAddGradȔ?o7Nb?!?G?Օ???"P
/gradient_tape/model/dense_8/BiasAdd/BiasAddGradBiasAddGrad?Zm??a?!?mCT???Q      Y@Y?0?0??a=??<ϳX@qa(?V?C@y>??z"??"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?92.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?38.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 