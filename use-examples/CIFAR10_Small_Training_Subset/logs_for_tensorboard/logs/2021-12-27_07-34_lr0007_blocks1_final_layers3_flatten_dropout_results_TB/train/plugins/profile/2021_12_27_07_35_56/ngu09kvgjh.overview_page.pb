?	?pZ??U?@?pZ??U?@!?pZ??U?@	0???v?0???v?!0???v?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9?pZ??U?@??F?0}??1??cZ+-?@I???v?"@Y?Z^??6??iU1?~B??r0*	Zd;?x?@2T
Iterator::Prefetch::Generatorp|??%@!?\m???W@)p|??%@1?\m???W@:Preprocessing2I
Iterator::PrefetchC8fٓ???!Q^????)C8fٓ???1Q^????:Preprocessing2?
SIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[1]::ForeverRepeat?4?($??!?vv?????)2=a????1?Ӝ#????:Preprocessing2O
Iterator::Root::Prefetch??V`????!?=?J?I??)??V`????1?=?J?I??:Preprocessing2?
MIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMap?)ʥ???!?n+>"???)ک??`???1??|????:Preprocessing2E
Iterator::Root?g?K6??!?[? ????)MM?7?Q??1ey??????:Preprocessing2V
Iterator::Root::Prefetch::Shard`?? @??!????r???)?6?X?O??1??j????:Preprocessing2_
(Iterator::Root::Prefetch::Shard::Rebatch?8?Z???!?mv:u???)???1>̎?1cbo
Z??:Preprocessing2s
<Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2=+i?7??!G"P?n???)=+i?7??1G"P?n???:Preprocessing2d
-Iterator::Root::Prefetch::Shard::Rebatch::Map?\???ʛ?!^??9p]??)e??]????1vV-?q???:Preprocessing2?
]Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceT? ?!ǆ?!?}? ???)T? ?!ǆ?1?}? ???:Preprocessing2x
AIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip??S?Ʊ?!?ʺ?W~??)?????w?1W????:Preprocessing2?
_Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor |(?r?!z?F?޸?) |(?r?1z?F?޸?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no90???v?I?>#????Q?K?$??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??F?0}????F?0}??!??F?0}??      ??!       "	??cZ+-?@??cZ+-?@!??cZ+-?@*      ??!       2      ??!       :	???v?"@???v?"@!???v?"@B      ??!       J	?Z^??6???Z^??6??!?Z^??6??R      ??!       Z	?Z^??6???Z^??6??!?Z^??6??b	U1?~B??U1?~B??!U1?~B??JGPUY0???v?b q?>#????y?K?$??X@?"n
@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??t^??!??t^??08"x
Jgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???']??!B2????08"R
2replica_1/model_1/model/block2a_expand_conv/Conv2DConv2Dks1?ޞ?!??R?????08"Q
3replica_1/model_1/model/block3a_project_conv/Conv2DConv2DL&B?h~?!??c????0"<
model_1/model/stem_conv/Conv2DConv2DOM,g9*~?!K??G???0"F
(replica_1/model_1/model/stem_conv/Conv2DConv2D$T?}?!@??????0"G
)model_1/model/block1a_project_conv/Conv2DConv2D?Z[Y??x?!??,????0"Q
3replica_1/model_1/model/block1a_project_conv/Conv2DConv2D?9??H?x?!?l?s?G??0"Q
3replica_1/model_1/model/block2a_project_conv/Conv2DConv2D)?.??u?!?'????0"D
+model_1/model/block2a_expand_activation/mulMulr*?~?'u?!\'YW???Q      Y@Y?7?z@a?Fv(lX@q???MYS@y???go?'?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?77.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 