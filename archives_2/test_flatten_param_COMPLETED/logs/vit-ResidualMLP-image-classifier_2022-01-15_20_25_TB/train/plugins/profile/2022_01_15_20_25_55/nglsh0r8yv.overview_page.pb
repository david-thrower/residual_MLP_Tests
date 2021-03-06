?	??????P@??????P@!??????P@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??????P@1	ȳ?2@I?·g	?H@r0*	X9???c@2T
Iterator::Root::ParallelMapV2????????!ť????9@)????????1ť????9@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??-???!=O9f?P9@)???f???1?9???5@:Preprocessing2E
Iterator::RootS??:??!F%???D@)?=???1???2??/@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice ;7m?i??!{Z???-@) ;7m?i??1{Z???-@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap8?*5{???!{????w:@)p??/ג?1z?<??'@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip"?:?vٷ?!???k?0M@)ݱ?&???1Y@?#&`@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor F?6?x?!?Upu@) F?6?x?1?Upu@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?73.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?kM??UR@Q P?J=?:@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "		ȳ?2@	ȳ?2@!	ȳ?2@*      ??!       2      ??!       :	?·g	?H@?·g	?H@!?·g	?H@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?kM??UR@y P?J=?:@?"N
2model_1/model/multi_head_attention_3/einsum/EinsumEinsum 83??t?! 83??t?"L
0model_1/model/multi_head_attention/einsum/EinsumEinsum?Ή`??t?!U??I?ń?"N
2model_1/model/multi_head_attention_2/einsum/EinsumEinsum?`t]??t?!???????"N
2model_1/model/multi_head_attention_1/einsum/EinsumEinsum?=?R?t?!D)?????"N
2model_1/model/multi_head_attention_4/einsum/EinsumEinsum(?6?6?t?!N?t??ҙ?"^
Bgradient_tape/model_1/model/multi_head_attention_1/einsum_1/EinsumEinsum?)?i?r?!????C???"^
Bgradient_tape/model_1/model/multi_head_attention_3/einsum_1/EinsumEinsumn]H2K?r?!?~?2????"^
Bgradient_tape/model_1/model/multi_head_attention_2/einsum_1/EinsumEinsum?"p?r?!Z_7???"\
@gradient_tape/model_1/model/multi_head_attention/einsum_1/EinsumEinsum?????r?!?<?Vc??"^
Bgradient_tape/model_1/model/multi_head_attention_4/einsum_1/EinsumEinsumYl?u??r?!AJ?ټ??I?v??????Q?C7P??X@YV???f???a8?e"?X@q???k A@y'?_c???"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?73.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?34.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 