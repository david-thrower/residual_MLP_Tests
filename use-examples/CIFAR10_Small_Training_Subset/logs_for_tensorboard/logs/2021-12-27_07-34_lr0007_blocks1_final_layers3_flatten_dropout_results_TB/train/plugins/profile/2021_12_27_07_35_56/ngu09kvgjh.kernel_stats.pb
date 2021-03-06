
?
?void cudnn::cnn::wgrad2d_grouped_direct_kernel<false, true, int, float, float, float>(cudnn::cnn::WgradGroupedDirectParams, float const*, float const*, float*, float, float)'*?2??8???E@???EH???EXb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad2d_grouped_direct_kernel<false, true, int, float, float, float>(cudnn::cnn::WgradGroupedDirectParams, float const*, float const*, float*, float, float)'*?2??8լ?E@լ?EHլ?EXbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2??8???@???H???Xb2replica_1/model_1/model/block2a_expand_conv/Conv2DhuMUB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b+model_1/model/block2a_expand_activation/mulhuZU?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b5replica_1/model_1/model/block2a_expand_activation/mulhuZU?B
?
?void implicit_convolve_sgemm<float, float, 512, 6, 8, 3, 3, 5, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)?R* 2Ւ8Ǻ?@Ǻ?HǺ?Xb2replica_1/model_1/model/block2a_expand_conv/Conv2DhuMUB
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b0model_1/model/block2a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b:replica_1/model_1/model/block2a_expand_bn/FusedBatchNormV3hu  ?B
?
?void precomputed_convolve_sgemm<float, 1024, 5, 5, 4, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)@?$*2??8ޑ?@ޑ?Hޑ?Xb2replica_1/model_1/model/block2a_expand_conv/Conv2DhuZU?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=?*2??8?կ@?կH?կXb2replica_1/model_1/model/block2a_expand_conv/Conv2DhuZU?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)K?2* 2Ւ8???@???H???Xb2replica_1/model_1/model/block2a_expand_conv/Conv2Dhu  HB
?
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8???@???H???Xb2replica_1/model_1/model/block2a_expand_conv/Conv2DhuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b$model_1/model/block2a_dwconv_pad/PadhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8ֈ?@ֈ?Hֈ?b.replica_1/model_1/model/block2a_dwconv_pad/PadhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block2a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8?ӡ@?ӡH?ӡb/model_1/model/block2a_expand_activation/SigmoidhuZU?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?ܳ@?ܳH?ܳb0replica_1/model_1/model/block3c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block3d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?׮@?׮H?׮b&model_1/model/block3f_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?ڭ@?ڭH?ڭb0replica_1/model_1/model/block3e_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?˭@?˭H?˭b&model_1/model/block3e_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block3g_dwconv/depthwisehu  ?B
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2??8???@???H???Xb2replica_1/model_1/model/block2a_expand_conv/Conv2DhuMUB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block3f_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8Ł?@Ł?HŁ?b0replica_1/model_1/model/block3g_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block3d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block3b_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?Ȥ@?ȤH?Ȥb&model_1/model/block3c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block5b_dwconv/depthwisehu  ?B
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2??8???@???H???Xb(model_1/model/block2a_expand_conv/Conv2DhuMUB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8ރ?@ރ?Hރ?b0replica_1/model_1/model/block5b_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8ʐ?@ʐ?Hʐ?b&model_1/model/block5d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block3b_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8݇?@݇?H݇?b0replica_1/model_1/model/block5i_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block5e_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?ː@?ːH?ːb&model_1/model/block5j_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block5c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8ڈ?@ڈ?Hڈ?b0replica_1/model_1/model/block5f_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block5i_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block5e_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block5d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block5h_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block5c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block5g_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?ׇ@?ׇH?ׇb0replica_1/model_1/model/block5j_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block5f_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?@?H?b&model_1/model/block5g_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block5h_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208ĝ?@ĝ?Hĝ?b0replica_1/model_1/model/block2a_dwconv/depthwisehu  ?B
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2??8???@???H???PXb2replica_1/model_1/model/block2a_expand_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b&model_1/model/block2a_dwconv/depthwisehu  ?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b+model_1/model/block3a_expand_activation/mulhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8Ǧ?@Ǧ?HǦ?b$model_1/model/block2d_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b+model_1/model/block2e_expand_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b.replica_1/model_1/model/block2d_activation/mulhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8爮@爮H爮b$model_1/model/block2c_activation/mulhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b$model_1/model/block2e_activation/mulhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8ǀ?@ǀ?Hǀ?b$model_1/model/block2b_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b.replica_1/model_1/model/block2f_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b+model_1/model/block2f_expand_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b+model_1/model/block2b_expand_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b.replica_1/model_1/model/block2g_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b.replica_1/model_1/model/block2e_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b+model_1/model/block2d_expand_activation/mulhuZU?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b5replica_1/model_1/model/block2f_expand_activation/mulhuZU?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b5replica_1/model_1/model/block2g_expand_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8?ح@?حH?حb.replica_1/model_1/model/block2b_activation/mulhuZU?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8?Э@?ЭH?Эb5replica_1/model_1/model/block2e_expand_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8?ƭ@?ƭH?ƭb.replica_1/model_1/model/block2c_activation/mulhuZU?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8?ŭ@?ŭH?ŭb5replica_1/model_1/model/block2c_expand_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8?­@?­H?­b+model_1/model/block2g_expand_activation/mulhuZU?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b5replica_1/model_1/model/block2d_expand_activation/mulhuZU?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b5replica_1/model_1/model/block3a_expand_activation/mulhuZU?B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b5replica_1/model_1/model/block2b_expand_activation/mulhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b$model_1/model/block2f_activation/mulhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8礭@礭H礭b$model_1/model/block2g_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b+model_1/model/block2c_expand_activation/mulhuZU?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?̧@?̧H?̧b&model_1/model/block3a_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b0replica_1/model_1/model/block3a_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8???@???H???b&model_1/model/block5a_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b0replica_1/model_1/model/block2b_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8?͚@?͚H?͚b0replica_1/model_1/model/block5a_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208퇙@퇙H퇙b0replica_1/model_1/model/block2c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b&model_1/model/block2b_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208뺒@뺒H뺒b0replica_1/model_1/model/block2e_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b&model_1/model/block2g_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b0replica_1/model_1/model/block2d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208?֏@?֏H?֏b&model_1/model/block2c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b0replica_1/model_1/model/block2f_dwconv/depthwisehu  ?B
?
?void precomputed_convolve_sgemm<float, 1024, 5, 5, 4, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)@?$*2??8ʍ?@ձ?H?ۇXb3replica_1/model_1/model/block1a_project_conv/Conv2DhuZU?B
?
?void precomputed_convolve_sgemm<float, 1024, 5, 5, 4, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)@?$*2??8???@龜H?чXb)model_1/model/block1a_project_conv/Conv2DhuZU?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208?ˎ@?ˎH?ˎb&model_1/model/block2f_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b&model_1/model/block2d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b&model_1/model/block2e_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b0replica_1/model_1/model/block2g_dwconv/depthwisehu  ?B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2Ւ8?ό@?όH?όb!model_1/model/stem_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2Ւ8蛌@蛌H蛌b+replica_1/model_1/model/stem_activation/mulhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2Ւ8ܐ?@ܐ?Hܐ?b$model_1/model/block1a_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2Ւ8???@???H???b.replica_1/model_1/model/block1a_activation/mulhuZU?B
t
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2??8???@??|H??|Xbmodel_1/model/stem_conv/Conv2DhuMUB
~
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2??8ߡ?@??zH??|Xb(replica_1/model_1/model/stem_conv/Conv2DhuMUB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b&model_1/model/block1a_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208???@???H???b0replica_1/model_1/model/block1a_dwconv/depthwisehu  ?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8???@???H???b-replica_1/model_1/model/block2b_se_excite/mulhuZU?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8???@???H???b-replica_1/model_1/model/block2d_se_excite/mulhuZU?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8???@???H???b-replica_1/model_1/model/block2c_se_excite/mulhuZU?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8ڗ?@ڗ?Hڗ?b-replica_1/model_1/model/block2e_se_excite/mulhuZU?B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8???@???H???b#model_1/model/block2f_se_excite/mulhuZU?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8???@???H???b-replica_1/model_1/model/block2f_se_excite/mulhuZU?B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8???@???H???b#model_1/model/block2g_se_excite/mulhuZU?B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8ϊ?@ϊ?Hϊ?b#model_1/model/block2b_se_excite/mulhuZU?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8و?@و?Hو?b-replica_1/model_1/model/block2g_se_excite/mulhuZU?B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8Ϛ?@Ϛ?HϚ?b#model_1/model/block2c_se_excite/mulhuZU?B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8???@???H???b#model_1/model/block2d_se_excite/mulhuZU?B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??	8???@???H???b#model_1/model/block2e_se_excite/mulhuZU?B
?
4ampere_scudnn_128x128_stridedB_splitK_interior_nn_v1???*?28???@???H???Xb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhuMUB
?
4ampere_scudnn_128x128_stridedB_splitK_interior_nn_v1???*?28???@???H???XbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhuMUB
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b#model_1/model/block1a_se_excite/mulhuZU?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b-replica_1/model_1/model/block1a_se_excite/mulhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ɸ?@ɸ?Hɸ?b0model_1/model/block3a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8Ȱ?@Ȱ?HȰ?b)model_1/model/block2b_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Ү?@Ү?HҮ?b.replica_1/model_1/model/block3a_dwconv_pad/PadhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8қ?@қ?Hқ?b:replica_1/model_1/model/block2e_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b:replica_1/model_1/model/block2g_expand_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b$model_1/model/block3a_dwconv_pad/PadhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b0model_1/model/block2c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b)model_1/model/block2f_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b:replica_1/model_1/model/block2d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b)model_1/model/block2c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b)model_1/model/block2d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b0model_1/model/block2b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b3replica_1/model_1/model/block2d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b0model_1/model/block2f_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ȯ?@ȯ?Hȯ?b)model_1/model/block2e_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b0model_1/model/block2d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b:replica_1/model_1/model/block2b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b0model_1/model/block2g_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b)model_1/model/block2g_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b3replica_1/model_1/model/block2g_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b:replica_1/model_1/model/block2f_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b:replica_1/model_1/model/block2c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8Ⱥ?@Ⱥ?HȺ?b0model_1/model/block2e_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b3replica_1/model_1/model/block2c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b3replica_1/model_1/model/block2f_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b3replica_1/model_1/model/block2b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b3replica_1/model_1/model/block2e_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block2d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b(model_1/model/block2f_activation/SigmoidhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b:replica_1/model_1/model/block3a_expand_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b2replica_1/model_1/model/block2g_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block2f_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block2b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b2replica_1/model_1/model/block2f_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Џ?@Џ?HЏ?b2replica_1/model_1/model/block2e_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block3a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block2c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b2replica_1/model_1/model/block2d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block2e_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block2g_expand_activation/SigmoidhuZU?B
j
redzone_checker*?2?@8ϥ?@??HæXb3replica_1/model_1/model/block3a_project_conv/Conv2Dh"uZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b2replica_1/model_1/model/block2b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Ϙ?@Ϙ?HϘ?b2replica_1/model_1/model/block2c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Ɔ?@Ɔ?HƆ?b/model_1/model/block3a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b(model_1/model/block2e_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b/model_1/model/block2c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b(model_1/model/block2d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b/model_1/model/block2f_expand_activation/SigmoidhuZU?B
?
?void wgrad_alg0_engine<float, 128, 6, 8, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int){?R* 2

8???@???H???Xb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b(model_1/model/block2b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8ƍ?@ƍ?Hƍ?b/model_1/model/block2g_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b(model_1/model/block2c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b/model_1/model/block2d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b(model_1/model/block2g_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b/model_1/model/block2b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b/model_1/model/block2e_expand_activation/SigmoidhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b$model_1/model/block2a_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b.replica_1/model_1/model/block2a_activation/mulhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2@8???@???H???b3replica_1/model_1/model/block1a_bn/FusedBatchNormV3hu  ?B
w
redzone_checker*?2?@8???@??H??Xb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh uZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2@8???@???H???b0replica_1/model_1/model/stem_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2@8???@???H???b)model_1/model/block1a_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2@8???@???H???b&model_1/model/stem_bn/FusedBatchNormV3hu  ?B
?
?void wgrad_alg0_engine<float, 128, 6, 8, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int){?R* 2

8???@???H???XbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhuMUB
?
redzone_checker*?2?@8臾@??H??XbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh uZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b%model_1/model/stem_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b2replica_1/model_1/model/block1a_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Ƚ?@Ƚ?HȽ?b/replica_1/model_1/model/stem_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8߳?@߳?H߳?b(model_1/model/block1a_activation/SigmoidhuZU?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, float, 128, 6, 8, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)?R* 2

8?Ǵ@?ǴH?ǴXb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhuMUB
?
?void cudnn::cnn::wgrad_alg1_engine<float, float, 128, 6, 8, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)?R* 2

8Ż?@Ż?HŻ?XbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhuMUB
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2P8?̭@??H??XbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2P8?˭@??H??Xb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void precomputed_convolve_sgemm<float, 1024, 5, 5, 4, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)@?$*2??8ܶ?@ܶ?Hܶ?Xbmodel_1/model/stem_conv/Conv2DhuZU?B
?
?void precomputed_convolve_sgemm<float, 1024, 5, 5, 4, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)@?$*2??8???@???H???Xb(replica_1/model_1/model/stem_conv/Conv2DhuZU?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=?*2??8?թ@?թH?թXbmodel_1/model/stem_conv/Conv2DhuZU?B
i
redzone_checker*?2?@8Ꭹ@??H??Xb2replica_1/model_1/model/block2a_expand_conv/Conv2DhuZU?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=?*2??8???@???H???Xb(replica_1/model_1/model/stem_conv/Conv2DhuZU?B
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8???@??SH??TXb3replica_1/model_1/model/block2a_project_conv/Conv2DhuMUB
]
redzone_checker*?2?@8???@??H??Xb&model_1/model/block1a_se_reduce/Conv2DhuZU?B
g
redzone_checker*?2?@8?ܥ@åH??Xb0replica_1/model_1/model/block1a_se_reduce/Conv2DhuZU?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b-replica_1/model_1/model/block2a_se_excite/mulhuZU?B
j
4ncclKernel_AllReduce_RING_LL_Sum_float(ncclWorkElem)`??*?28֤?@??'H??sbncclAllReducehu  B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=?*2??8?ٚ@?ٚH?ٚXb3replica_1/model_1/model/block2a_project_conv/Conv2DhuZU?B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b#model_1/model/block2a_se_excite/mulhuZU?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2?	8???@Ӻ2H??2Xb3replica_1/model_1/model/block3a_project_conv/Conv2Dhu  ?B
?
%ampere_scudnn_128x32_relu_small_nn_v1??**@2?"8???@???H???Xb3replica_1/model_1/model/block2a_project_conv/Conv2DhuMUB
?
?void precomputed_convolve_sgemm<float, 1024, 5, 5, 4, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)@?$*2??8???@???H???Xb3replica_1/model_1/model/block2a_project_conv/Conv2DhuZU?B
q
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8???@???H???Xbmodel_1/model/stem_conv/Conv2DhuMUB
{
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8???@???H???Xb(replica_1/model_1/model/stem_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2??8???@???H???Xb)model_1/model/block1a_project_conv/Conv2DhuMUB
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8Ѯ?@Ѯ?HѮ?b)model_1/model/block2a_bn/FusedBatchNormV3hu  ?B
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2??8???@???H???Xb3replica_1/model_1/model/block1a_project_conv/Conv2DhuMUB
_
redzone_checker*?2?@8???@??H??Xb(replica_1/model_1/model/stem_conv/Conv2DhuZU?B
U
redzone_checker*?2?@8?ٌ@??H??Xbmodel_1/model/stem_conv/Conv2DhuZU?B

&ampere_scudnn_128x64_relu_medium_nn_v1???*?2??8?Ɗ@?ƊH?ƊXb)model_1/model/block1a_project_conv/Conv2DhuMUB
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8???@???H???b3replica_1/model_1/model/block2a_bn/FusedBatchNormV3hu  ?B
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2??8ֈ?@ֈ?Hֈ?Xb3replica_1/model_1/model/block1a_project_conv/Conv2DhuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8?щ@?щH?щb(model_1/model/block2a_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b2replica_1/model_1/model/block2a_activation/SigmoidhuZU?B
|
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8???@???H???Xb)model_1/model/block1a_project_conv/Conv2DhuMUB
?
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8?ˈ@?ˈH?ˈXb3replica_1/model_1/model/block1a_project_conv/Conv2DhuMUB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=?*2??8???@???H???Xb3replica_1/model_1/model/block1a_project_conv/Conv2DhuZU?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=?*2??8???@???H???Xb)model_1/model/block1a_project_conv/Conv2DhuZU?B
y
+ampere_scudnn_128x64_relu_xregs_large_nn_v1???*?2??8?Ї@?ЇH?ЇXbmodel_1/model/stem_conv/Conv2Dhu  ?A
h
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???bmodel_1/model/block1c_add/addhuZU?B
?
+ampere_scudnn_128x64_relu_xregs_large_nn_v1???*?2??8Խ?@Խ?HԽ?Xb(replica_1/model_1/model/stem_conv/Conv2Dhu  ?A
h
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8ε?@ε?Hε?bmodel_1/model/block1d_add/addhuZU?B
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b'replica_1/model_1/model/block1d_add/addhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b$model_1/model/block1c_activation/mulhuZU?B
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8դ?@դ?Hդ?b'replica_1/model_1/model/block1c_add/addhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b$model_1/model/block1b_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b.replica_1/model_1/model/block1c_activation/mulhuZU?B
h
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???bmodel_1/model/block1b_add/addhuZU?B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b$model_1/model/block1d_activation/mulhuZU?B
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b'replica_1/model_1/model/block1b_add/addhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b.replica_1/model_1/model/block1b_activation/mulhuZU?B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8???@???H???b.replica_1/model_1/model/block1d_activation/mulhuZU?B
t
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2??8???@???H???Xbmodel_1/model/stem_conv/Conv2DhuMUB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8???@???H???PXb2replica_1/model_1/model/block2c_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8???@???H???PXb2replica_1/model_1/model/block2e_expand_conv/Conv2Dh
g
redzone_checker*?2?@8???@??H??Xb0replica_1/model_1/model/block2a_se_reduce/Conv2DhuZU?B
~
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2??8???@???H???Xb(replica_1/model_1/model/stem_conv/Conv2DhuMUB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8?ɀ@?ɀH?ɀPXb2replica_1/model_1/model/block2b_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8?Ā@?ĀH?ĀPXb(model_1/model/block2g_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??@??H??PXb2replica_1/model_1/model/block2f_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??@??H??PXb(model_1/model/block3a_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??@??H??PXb2replica_1/model_1/model/block2d_expand_conv/Conv2Dh
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b5replica_1/model_1/model/block3e_expand_activation/mulhuZU?B
\
redzone_checker*?2?@8??~@H??Xb&model_1/model/block1a_se_expand/Conv2DhuZU?B
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??~@??~H??~PXb(model_1/model/block2c_expand_conv/Conv2Dh
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b$model_1/model/block3e_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b.replica_1/model_1/model/block3e_activation/mulhuZU?B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b$model_1/model/block3f_activation/mulhuZU?B
f
redzone_checker*?2?@8Ѳ~@??H??Xb0replica_1/model_1/model/block1a_se_expand/Conv2DhuZU?B
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??~@??~H??~PXb(model_1/model/block2b_expand_conv/Conv2Dh
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b5replica_1/model_1/model/block3f_expand_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b.replica_1/model_1/model/block3f_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8Ѣ~@Ѣ~HѢ~b.replica_1/model_1/model/block3g_activation/mulhuZU?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b+model_1/model/block3b_expand_activation/mulhuZU?B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b$model_1/model/block3b_activation/mulhuZU?B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8˙~@˙~H˙~b$model_1/model/block3d_activation/mulhuZU?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b+model_1/model/block3f_expand_activation/mulhuZU?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b+model_1/model/block3g_expand_activation/mulhuZU?B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b5replica_1/model_1/model/block3b_expand_activation/mulhuZU?B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b5replica_1/model_1/model/block3c_expand_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b.replica_1/model_1/model/block3c_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b.replica_1/model_1/model/block3b_activation/mulhuZU?B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b$model_1/model/block3g_activation/mulhuZU?B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b5replica_1/model_1/model/block3g_expand_activation/mulhuZU?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b+model_1/model/block3c_expand_activation/mulhuZU?B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b$model_1/model/block3c_activation/mulhuZU?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b.replica_1/model_1/model/block3d_activation/mulhuZU?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??~@??~H??~b+model_1/model/block3d_expand_activation/mulhuZU?B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??}@??}H??}b5replica_1/model_1/model/block3d_expand_activation/mulhuZU?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??}@??}H??}b+model_1/model/block4a_expand_activation/mulhuZU?B
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??}@??}H??}PXb2replica_1/model_1/model/block2g_expand_conv/Conv2Dh
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??}@??}H??}b+model_1/model/block3e_expand_activation/mulhuZU?B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??}@??}H??}b5replica_1/model_1/model/block4a_expand_activation/mulhuZU?B
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??}@??}H??}PXb(model_1/model/block2e_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8˼}@˼}H˼}PXb(model_1/model/block2d_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??}@??}H??}PXb2replica_1/model_1/model/block3a_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?"8??|@??|H??|PXb(model_1/model/block2f_expand_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??{@??{H??{b0replica_1/model_1/model/block1c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208ʕy@ʕyHʕyb&model_1/model/block1d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??y@??yH??yb&model_1/model/block1b_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208ʎy@ʎyHʎyb&model_1/model/block1c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??x@??xH??xb0replica_1/model_1/model/block1d_dwconv/depthwisehu  ?B
?
?void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 16> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, xmma_cudnn::Row, 128, 16> >, true, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 16> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, xmma_cudnn::Row, 128, 16> >, true, 4>::Params)? ??*?28??x@??5H??CPXb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??x@??xH??xb0replica_1/model_1/model/block1b_dwconv/depthwisehu  ?B
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??w@??;H??;Xb3replica_1/model_1/model/block3a_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??w@??wH??wXb3replica_1/model_1/model/block2b_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??v@??vH??vXb)model_1/model/block2f_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??v@??vH??vXb3replica_1/model_1/model/block2d_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8ɀv@ɀvHɀvXb)model_1/model/block2d_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??u@??uH??uXb)model_1/model/block2b_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??u@??uH??uXb)model_1/model/block2g_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??u@??uH??uXb3replica_1/model_1/model/block2f_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??u@??uH??uXb)model_1/model/block2e_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??u@??uH??uXb3replica_1/model_1/model/block2c_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??u@??uH??uXb)model_1/model/block2c_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??u@??uH??uXb3replica_1/model_1/model/block2e_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??u@??uH??uXb3replica_1/model_1/model/block2g_project_conv/Conv2DhuMUB
?
?void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 16> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, xmma_cudnn::Row, 128, 16> >, true, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 16> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, xmma_cudnn::Row, 128, 16> >, true, 4>::Params)? ??*?28??u@??6Hث?PXbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
_
redzone_checker*?2?@8Ɍt@??H??Xb)model_1/model/block1a_project_conv/Conv2DhuZU?B
i
redzone_checker*?2?@8??s@©H??Xb3replica_1/model_1/model/block1a_project_conv/Conv2DhuZU?B
i
redzone_checker*?2?@8??r@??H??Xb3replica_1/model_1/model/block2a_project_conv/Conv2DhuZU?B
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??m@??mH??mb2replica_1/model_1/model/block1b_drop/dropout/Mul_1huZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??l@??lH??lb-replica_1/model_1/model/block1c_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8ʺl@ʺlHʺlb-replica_1/model_1/model/block1b_se_excite/mulhuZU?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??k@??kH??kb(model_1/model/block1d_drop/dropout/Mul_1huZU?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??k@??kH??kb(model_1/model/block1c_drop/dropout/Mul_1huZU?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??k@??kH??kb(model_1/model/block1b_drop/dropout/Mul_1huZU?B
?
?void implicit_convolve_sgemm<float, float, 512, 6, 8, 3, 3, 5, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)?R* 2?8??k@??kH??kXb3replica_1/model_1/model/block3a_project_conv/Conv2DhuMUB
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??k@??kH??kb2replica_1/model_1/model/block1d_drop/dropout/Mul_1huZU?B
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??k@??kH??kb2replica_1/model_1/model/block1c_drop/dropout/Mul_1huZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??k@??kH??kb#model_1/model/block1d_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??j@??jH??jb#model_1/model/block1c_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??j@??jH??jb#model_1/model/block1b_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??j@??jH??jb-replica_1/model_1/model/block1d_se_excite/mulhuZU?B
f
redzone_checker*?2?@8??f@??H??Xb0replica_1/model_1/model/block2a_se_expand/Conv2DhuZU?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8??d@??dH??db&model_1/model/block6a_dwconv/depthwisehu  ?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??c@??cH??cb-replica_1/model_1/model/block3b_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??c@??cH??cb-replica_1/model_1/model/block3c_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??c@??cH??cb#model_1/model/block3g_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??b@??bH??bb-replica_1/model_1/model/block3e_se_excite/mulhuZU?B
z
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb(model_1/model/block3c_expand_conv/Conv2DhuMUB
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb2replica_1/model_1/model/block3c_expand_conv/Conv2DhuMUB
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??b@??bH??bb#model_1/model/block3f_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??b@??bH??bb#model_1/model/block3e_se_excite/mulhuZU?B
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb2replica_1/model_1/model/block4a_expand_conv/Conv2DhuMUB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??b@??bH??bb-replica_1/model_1/model/block3d_se_excite/mulhuZU?B
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb2replica_1/model_1/model/block3b_expand_conv/Conv2DhuMUB
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb2replica_1/model_1/model/block3f_expand_conv/Conv2DhuMUB
z
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb(model_1/model/block3b_expand_conv/Conv2DhuMUB
z
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb(model_1/model/block3f_expand_conv/Conv2DhuMUB
z
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb(model_1/model/block3d_expand_conv/Conv2DhuMUB
z
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb(model_1/model/block4a_expand_conv/Conv2DhuMUB
z
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb(model_1/model/block3g_expand_conv/Conv2DhuMUB
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb2replica_1/model_1/model/block3g_expand_conv/Conv2DhuMUB
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb2replica_1/model_1/model/block3e_expand_conv/Conv2DhuMUB
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb2replica_1/model_1/model/block3d_expand_conv/Conv2DhuMUB
z
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8??b@??bH??bXb(model_1/model/block3e_expand_conv/Conv2DhuMUB
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??b@??bH??bb#model_1/model/block3d_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??b@??bH??bb#model_1/model/block3c_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??b@??bH??bb-replica_1/model_1/model/block3f_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8Ɖb@ƉbHƉbb-replica_1/model_1/model/block3g_se_excite/mulhuZU?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab%model_1/model/block2b_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab/replica_1/model_1/model/block2b_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab/replica_1/model_1/model/block2d_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab/replica_1/model_1/model/block2e_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab%model_1/model/block2c_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab/replica_1/model_1/model/block2c_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab/replica_1/model_1/model/block2g_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab/replica_1/model_1/model/block2f_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab%model_1/model/block2f_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab%model_1/model/block2g_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab%model_1/model/block2d_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?88??a@??aH??ab%model_1/model/block2e_se_squeeze/Meanhu  ?B
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??a@??aH??aXb3replica_1/model_1/model/block3b_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??a@??aH??aXb)model_1/model/block3g_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??a@??aH??aXb)model_1/model/block3f_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??a@??aH??aXb)model_1/model/block3e_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??a@??aH??aXb3replica_1/model_1/model/block3c_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??`@??`H??`Xb3replica_1/model_1/model/block3d_project_conv/Conv2DhuMUB
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??`@??`H??`b#model_1/model/block3b_se_excite/mulhuZU?B
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??`@??`H??`Xb)model_1/model/block3c_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??`@??`H??`Xb)model_1/model/block3d_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??`@??`H??`Xb3replica_1/model_1/model/block3e_project_conv/Conv2DhuMUB
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??`@??`H??`b;replica_1/model_1/model/block1c_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??`@??`H??`b3replica_1/model_1/model/block1b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??`@??`H??`b;replica_1/model_1/model/block1d_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??`@??`H??`b3replica_1/model_1/model/block1d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??`@??`H??`b;replica_1/model_1/model/block1b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??`@??`H??`b3replica_1/model_1/model/block1c_bn/FusedBatchNormV3hu  ?B
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??_@??_H??_Xb3replica_1/model_1/model/block3g_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??_@??_H??_Xb)model_1/model/block3b_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??_@??_H??_Xb3replica_1/model_1/model/block3f_project_conv/Conv2DhuMUB
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??_@??_H??_b1model_1/model/block1c_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??_@??_H??_b1model_1/model/block1b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??_@??_H??_b)model_1/model/block1d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??_@??_H??_b;replica_1/model_1/model/block1a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??^@??^H??^b1model_1/model/block1d_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??^@??^H??^b)model_1/model/block1c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??^@??^H??^b)model_1/model/block1b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2 8??^@??^H??^b1model_1/model/block1a_project_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, -1, -1, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)"*?2`8??]@??]H??]b0replica_1/model_1/model/block6a_dwconv/depthwisehu  ?B
?
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8??\@??\H??\Xb3replica_1/model_1/model/block1d_project_conv/Conv2DhuMUB
?
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8??\@??\H??\Xb3replica_1/model_1/model/block1c_project_conv/Conv2DhuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??\@??\H??\b2replica_1/model_1/model/block1c_activation/SigmoidhuZU?B
y
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8??\@??\H??\Xb)model_1/model/block1c_project_conv/Conv2DhuMUB
y
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8??\@??\H??\Xb)model_1/model/block1d_project_conv/Conv2DhuMUB
y
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8??\@??\H??\Xb)model_1/model/block1b_project_conv/Conv2DhuMUB
?
%ampere_scudnn_128x32_relu_small_nn_v1??**@2??8Ĺ\@Ĺ\HĹ\Xb3replica_1/model_1/model/block1b_project_conv/Conv2DhuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??\@??\H??\b(model_1/model/block1b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??\@??\H??\b2replica_1/model_1/model/block1b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??\@??\H??\b(model_1/model/block1d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??[@??[H??[b(model_1/model/block1c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8ä[@ä[Hä[b2replica_1/model_1/model/block1d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??[@??[H??[b$model_1/model/block4a_dwconv_pad/PadhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5e_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb5replica_1/model_1/model/block5g_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb$model_1/model/block5h_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb5replica_1/model_1/model/block5e_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb$model_1/model/block5e_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb+model_1/model/block5e_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb+model_1/model/block6a_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb5replica_1/model_1/model/block5b_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb+model_1/model/block5j_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb$model_1/model/block5c_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5i_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb5replica_1/model_1/model/block6a_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb5replica_1/model_1/model/block5c_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5g_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb5replica_1/model_1/model/block5f_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb+model_1/model/block5h_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb$model_1/model/block5i_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb$model_1/model/block5g_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb+model_1/model/block5g_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb5replica_1/model_1/model/block5d_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb+model_1/model/block5d_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb$model_1/model/block5d_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8߸Z@߸ZH߸Zb+model_1/model/block5f_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb+model_1/model/block5b_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5h_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb$model_1/model/block5j_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5d_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb5replica_1/model_1/model/block5j_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5b_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8߯Z@߯ZH߯Zb+model_1/model/block5i_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8߮Z@߮ZH߮Zb$model_1/model/block5b_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5j_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb+model_1/model/block5c_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5f_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8ìZ@ìZHìZb5replica_1/model_1/model/block5h_expand_activation/mulhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??Z@??ZH??Zb.replica_1/model_1/model/block4a_dwconv_pad/PadhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb$model_1/model/block5f_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8??Z@??ZH??Zb.replica_1/model_1/model/block5c_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?\8äZ@äZHäZb5replica_1/model_1/model/block5i_expand_activation/mulhuZU?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2?8??Z@??ZH??ZXb2replica_1/model_1/model/block2a_expand_conv/Conv2Dhu  ?B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??Y@??YH??YXb(model_1/model/block7d_expand_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??Y@??YH??YXb(model_1/model/block7c_expand_conv/Conv2Dh
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?"8??X@??XH??XXb3replica_1/model_1/model/block2a_project_conv/Conv2DhuMUB
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??X@??XH??XXb2replica_1/model_1/model/block7d_expand_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??X@??XH??XXb(model_1/model/block7b_expand_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??W@??WH??WXb2replica_1/model_1/model/block7b_expand_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8W@WHWXb2replica_1/model_1/model/block7c_expand_conv/Conv2Dh
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?8??V@??VH??Vb%model_1/model/block1a_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?8??V@??VH??Vb/replica_1/model_1/model/block1a_se_squeeze/Meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb9replica_1/model_1/model/block3b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb9replica_1/model_1/model/block3d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8ޯV@ޯVHޯVb(model_1/model/block3d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb/model_1/model/block4a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8¨V@¨VH¨Vb2replica_1/model_1/model/block3e_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb2replica_1/model_1/model/block3d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8¦V@¦VH¦Vb2replica_1/model_1/model/block3g_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb2replica_1/model_1/model/block3b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb9replica_1/model_1/model/block3e_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb2replica_1/model_1/model/block3c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb(model_1/model/block3c_activation/SigmoidhuZU?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)=?*2?"8??V@??VH??VXb3replica_1/model_1/model/block3a_project_conv/Conv2DhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??V@??VH??Vb/model_1/model/block3c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub/model_1/model/block3e_expand_activation/SigmoidhuZU?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??U@??UH??Ub&model_1/model/block4b_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub2replica_1/model_1/model/block3f_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub(model_1/model/block3b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub9replica_1/model_1/model/block4a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub/model_1/model/block3f_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub/model_1/model/block3g_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub9replica_1/model_1/model/block3f_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub(model_1/model/block3g_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub9replica_1/model_1/model/block3g_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub9replica_1/model_1/model/block3c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub/model_1/model/block3b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub/model_1/model/block3d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub(model_1/model/block3f_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??U@??UH??Ub(model_1/model/block3e_activation/SigmoidhuZU?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??U@??UH??Ub0replica_1/model_1/model/block4f_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??U@??UH??Ub0replica_1/model_1/model/block4g_dwconv/depthwisehu  ?B
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?"8??T@??TH??TXb)model_1/model/block2a_project_conv/Conv2DhuMUB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb&model_1/model/block4e_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb&model_1/model/block4d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb&model_1/model/block4f_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb&model_1/model/block4c_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb0replica_1/model_1/model/block4i_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb0replica_1/model_1/model/block4h_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb:replica_1/model_1/model/block3g_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb0model_1/model/block3f_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb:replica_1/model_1/model/block3b_expand_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb&model_1/model/block4j_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb0model_1/model/block3e_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb0model_1/model/block3c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb:replica_1/model_1/model/block3f_expand_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb&model_1/model/block4h_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb0model_1/model/block3d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb:replica_1/model_1/model/block3e_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ݒT@ݒTHݒTb)model_1/model/block3e_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb:replica_1/model_1/model/block3c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb:replica_1/model_1/model/block4a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb3replica_1/model_1/model/block3f_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb&model_1/model/block4i_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb&model_1/model/block4g_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb0model_1/model/block3g_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb0model_1/model/block4a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb)model_1/model/block3b_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb0replica_1/model_1/model/block4j_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb3replica_1/model_1/model/block3g_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb)model_1/model/block3d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??T@??TH??Tb)model_1/model/block3f_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8܂T@܂TH܂Tb)model_1/model/block3g_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8݁T@݁TH݁Tb0model_1/model/block3b_expand_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??T@??TH??Tb0replica_1/model_1/model/block4d_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??S@??SH??Sb0replica_1/model_1/model/block4b_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??S@??SH??Sb:replica_1/model_1/model/block3d_expand_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??S@??SH??Sb0replica_1/model_1/model/block4c_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??S@??SH??Sb3replica_1/model_1/model/block3c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??S@??SH??Sb)model_1/model/block3c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??S@??SH??Sb3replica_1/model_1/model/block3b_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??S@??SH??Sb0replica_1/model_1/model/block4e_dwconv/depthwisehu  ?B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??S@??SH??SXb)model_1/model/block7b_project_conv/Conv2Dh
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??S@??SH??Sb3replica_1/model_1/model/block3d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??S@??SH??Sb3replica_1/model_1/model/block3e_bn/FusedBatchNormV3hu  ?B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??S@??SH??SXb)model_1/model/block7c_project_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??S@??SH??SXb)model_1/model/block7d_project_conv/Conv2Dh
?
?void precomputed_convolve_sgemm<float, 1024, 5, 5, 4, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)@?$*2?"8??S@??SH??SXb3replica_1/model_1/model/block3a_project_conv/Conv2DhuZU?B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??S@??SH??SXb3replica_1/model_1/model/block7d_project_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??S@??SH??SXb3replica_1/model_1/model/block7c_project_conv/Conv2Dh
?
%ampere_scudnn_128x32_relu_small_nn_v1??**@2?8??R@??RH??RXb3replica_1/model_1/model/block3a_project_conv/Conv2DhuMUB
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??R@??RH??RXb3replica_1/model_1/model/block7b_project_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ??*?2
8??Q@??QH??QXb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3>(cutlass_tensorop_s1688wgrad_analytic_tf32_128x256_16x3::Params)? ??*?2
8??P@??PH??PXbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, 3, 3, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)6 ?:*
208??N@??NH??Nb&model_1/model/block7c_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, 3, 3, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)6 ?:*
208??N@??NH??Nb&model_1/model/block7d_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, 3, 3, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)6 ?:*
208??M@??MH??Mb0replica_1/model_1/model/block7c_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, 3, 3, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)6 ?:*
208??M@??MH??Mb&model_1/model/block7b_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, 3, 3, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)6 ?:*
208??M@??MH??Mb0replica_1/model_1/model/block7d_dwconv/depthwisehu??EB
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_256x64_16x4>(cutlass_tensorop_s1688wgrad_analytic_tf32_256x64_16x4::Params)? ??*?2(8??M@??MH??MXb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, 3, 3, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)6 ?:*
208??M@??MH??Mb0replica_1/model_1/model/block7b_dwconv/depthwisehu??EB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?M8??K@??KH??Kb.replica_1/model_1/model/block3a_activation/mulhuZU?B
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??K@??KH??KXb)model_1/model/block4d_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??K@??KH??KXb)model_1/model/block4e_project_conv/Conv2DhuMUB
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?M8??K@??KH??Kb$model_1/model/block3a_activation/mulhuZU?B
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??K@??KH??KXb)model_1/model/block4b_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8ڳK@ڳKHڳKXb)model_1/model/block4c_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??K@??KH??KXb)model_1/model/block4h_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??K@??KH??KXb)model_1/model/block4f_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8ڥK@ڥKHڥKXb)model_1/model/block4j_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??K@??KH??KXb)model_1/model/block4i_project_conv/Conv2DhuMUB
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??K@??KH??KXb)model_1/model/block4g_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??K@??KH??KXb3replica_1/model_1/model/block4f_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8ݚK@ݚKHݚKXb3replica_1/model_1/model/block4e_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??J@??JH??JXb3replica_1/model_1/model/block4g_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??J@??JH??JXb3replica_1/model_1/model/block4h_project_conv/Conv2DhuMUB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??J@??JH??JPXb(model_1/model/block5b_expand_conv/Conv2Dh
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??J@??JH??JXb3replica_1/model_1/model/block4d_project_conv/Conv2DhuMUB
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688wgrad_analytic_tf32_256x64_16x4>(cutlass_tensorop_s1688wgrad_analytic_tf32_256x64_16x4::Params)? ??*?2(8??J@??JH??JXbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??J@??JH??JXb3replica_1/model_1/model/block4b_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??J@??JH??JXb3replica_1/model_1/model/block4j_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??J@??JH??JXb3replica_1/model_1/model/block4i_project_conv/Conv2DhuMUB
?
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??J@??JH??JXb3replica_1/model_1/model/block4c_project_conv/Conv2DhuMUB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??J@??JH??JPXb(model_1/model/block5c_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??J@??JH??JPXb2replica_1/model_1/model/block5b_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??J@??JH??JPXb(model_1/model/block5d_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??J@??JH??JPXb(model_1/model/block5j_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??J@??JH??JPXb2replica_1/model_1/model/block5c_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??J@??JH??JPXb2replica_1/model_1/model/block5d_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8ڎJ@ڎJHڎJPXb(model_1/model/block5e_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??I@??IH??IPXb(model_1/model/block5f_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8ݹI@ݹIHݹIPXb2replica_1/model_1/model/block5f_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8ݸI@ݸIHݸIPXb2replica_1/model_1/model/block5i_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8ݳI@ݳIHݳIPXb2replica_1/model_1/model/block5e_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??I@??IH??IPXb2replica_1/model_1/model/block5g_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??I@??IH??IPXb(model_1/model/block5g_expand_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??I@??IH??Ib&model_1/model/block6k_dwconv/depthwisehu??EB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??I@??IH??IPXb(model_1/model/block6a_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??I@??IH??IPXb2replica_1/model_1/model/block5j_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??I@??IH??IPXb(model_1/model/block5h_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??I@??IH??IPXb(model_1/model/block5i_expand_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??I@??IH??Ib&model_1/model/block6e_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6m_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6c_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6h_dwconv/depthwisehu??EB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??H@??HH??HPXb2replica_1/model_1/model/block6a_expand_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6i_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6i_dwconv/depthwisehu??EB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??H@??HH??HPXb2replica_1/model_1/model/block5h_expand_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6d_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6h_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6l_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6g_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6f_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6g_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6j_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6b_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6l_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6m_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6k_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6j_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb&model_1/model/block6f_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6d_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208ܻH@ܻHHܻHb0replica_1/model_1/model/block6e_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6c_dwconv/depthwisehu??EB
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, -1, -1, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)* ?H*
208??H@??HH??Hb0replica_1/model_1/model/block6b_dwconv/depthwisehu??EB
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5b_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb-replica_1/model_1/model/block5b_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5i_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5e_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb-replica_1/model_1/model/block5c_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5d_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5c_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb-replica_1/model_1/model/block5d_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5j_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8ۢF@ۢFHۢFb-replica_1/model_1/model/block5e_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8۟F@۟FH۟Fb-replica_1/model_1/model/block5i_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8ۜF@ۜFHۜFb-replica_1/model_1/model/block5j_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb-replica_1/model_1/model/block5h_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb-replica_1/model_1/model/block5f_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5f_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5h_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb-replica_1/model_1/model/block5g_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??F@??FH??Fb#model_1/model/block5g_se_excite/mulhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??E@??EH??Eb$model_1/model/block6a_dwconv_pad/PadhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??E@??EH??Eb.replica_1/model_1/model/block6a_dwconv_pad/PadhuZU?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)K?2* 2?8۰E@۰EH۰EXb3replica_1/model_1/model/block3a_project_conv/Conv2Dhu  HB
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?%8??A@??AH??Ab/replica_1/model_1/model/block2a_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?%8??A@??AH??Ab%model_1/model/block2a_se_squeeze/Meanhu  ?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??A@??AH??Ab5replica_1/model_1/model/block7b_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8֑A@֑AH֑Ab+model_1/model/block4f_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??A@??AH??Ab5replica_1/model_1/model/block4i_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8ًA@ًAHًAb.replica_1/model_1/model/block4f_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??A@??AH??Ab$model_1/model/block4e_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??A@??AH??Ab5replica_1/model_1/model/block4e_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??A@??AH??Ab+model_1/model/block4e_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8فA@فAHفAb.replica_1/model_1/model/block7c_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block4g_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block4b_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block4d_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block4j_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block4f_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block7b_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block4b_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block4d_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block4h_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block5a_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block4d_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block5a_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block7c_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block7d_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block4b_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block4c_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block4c_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block4c_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block4f_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block4h_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block4d_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block4g_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block7b_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block4j_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block4b_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block4j_expand_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block7d_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block7d_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block4h_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block4i_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block4c_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block7d_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block4g_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block7c_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block4e_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block4i_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block4j_expand_activation/mulhuZU?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b+model_1/model/block7c_expand_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block5a_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block7b_activation/mulhuZU?B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b5replica_1/model_1/model/block5a_expand_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block4i_activation/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b.replica_1/model_1/model/block4h_activation/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?B8??@@??@H??@b$model_1/model/block4g_activation/mulhuZU?B
?
?void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 16> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, xmma_cudnn::Row, 128, 16> >, true, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 16> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, xmma_cudnn::Row, 128, 16> >, true, 4>::Params)? ??*?28ֶ@@ֶ@Hֶ@PXb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8???@???H???b9replica_1/model_1/model/block5c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b2replica_1/model_1/model/block5j_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b9replica_1/model_1/model/block5i_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b9replica_1/model_1/model/block5h_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b9replica_1/model_1/model/block5f_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b2replica_1/model_1/model/block5c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b(model_1/model/block5d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8ؓ>@ؓ>Hؓ>b9replica_1/model_1/model/block5d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Փ>@Փ>HՓ>b/model_1/model/block5f_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b9replica_1/model_1/model/block5j_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b/model_1/model/block5e_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b(model_1/model/block5h_activation/SigmoidhuZU?B
?
?void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 16> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, xmma_cudnn::Row, 128, 16> >, true, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 16> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_tf32_traits<unsigned int, float>, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 16, 2, 2, 1, 1>, true, 16, xmma_cudnn::Row, 128, 16> >, true, 4>::Params)? ??*?28??>@??>H??>PXbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
?
?_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi256ELi64ELi16EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi256ELi16EEENS_10tfloat32_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi256ELi16EEELi128ENSG_ILi8ELi4EEELi4EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi32ELi32EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi16ELi64EEESC_NSE_INSG_ILi64ELi16EEELi128ESI_Li4EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi32ELi32EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi16EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi8EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi64ELi8ELi4ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE? ??*?2(8??>@??>H??>Xb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b9replica_1/model_1/model/block6a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??>@??>H??>b9replica_1/model_1/model/block5b_expand_activation/SigmoidhuZU?B
?
&ampere_scudnn_128x64_relu_medium_nn_v1???*?2?8؀>@؀>H؀>Xb3replica_1/model_1/model/block3a_project_conv/Conv2DhuMUB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b(model_1/model/block5f_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b/model_1/model/block5g_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b(model_1/model/block5b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b2replica_1/model_1/model/block5b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b/model_1/model/block5i_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b/model_1/model/block5b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b/model_1/model/block5d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b2replica_1/model_1/model/block5h_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b/model_1/model/block5c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b/model_1/model/block5j_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b2replica_1/model_1/model/block5i_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b2replica_1/model_1/model/block5e_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b9replica_1/model_1/model/block5e_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b(model_1/model/block5j_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b/model_1/model/block5h_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b2replica_1/model_1/model/block5d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b2replica_1/model_1/model/block5f_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b9replica_1/model_1/model/block5g_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b/model_1/model/block6a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b(model_1/model/block5e_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b(model_1/model/block5c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b(model_1/model/block5g_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b2replica_1/model_1/model/block5g_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??=@??=H??=b(model_1/model/block5i_activation/SigmoidhuZU?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5e_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5f_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb)model_1/model/block5g_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5j_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb)model_1/model/block5e_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb)model_1/model/block5i_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b)model_1/model/block5d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5b_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block5b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5d_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5i_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5c_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b)model_1/model/block5i_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5g_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb3replica_1/model_1/model/block5h_project_conv/Conv2Dhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb)model_1/model/block5d_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5h_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b3replica_1/model_1/model/block5b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb)model_1/model/block5j_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b)model_1/model/block5b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block5j_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block5g_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block5c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b)model_1/model/block5h_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b)model_1/model/block5e_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb)model_1/model/block5h_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5e_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8տ<@տ<Hտ<Xb)model_1/model/block5c_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block5e_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8ؾ<@ؾ<Hؾ<b3replica_1/model_1/model/block5e_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b3replica_1/model_1/model/block5i_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b)model_1/model/block5g_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5f_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5j_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b)model_1/model/block5c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b3replica_1/model_1/model/block5c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block5d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5g_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block5i_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b3replica_1/model_1/model/block5j_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b3replica_1/model_1/model/block5d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b)model_1/model/block5f_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8ջ<@ջ<Hջ<b0model_1/model/block5f_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8ջ<@ջ<Hջ<b)model_1/model/block5j_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8ջ<@ջ<Hջ<Xb)model_1/model/block5f_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block6a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8׺<@׺<H׺<b3replica_1/model_1/model/block5g_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b3replica_1/model_1/model/block5h_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.*8??<@??<H??<Xb)model_1/model/block5b_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block6a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b0model_1/model/block5h_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8??<@??<H??<b:replica_1/model_1/model/block5i_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?
8ش<@ش<Hش<b3replica_1/model_1/model/block5f_bn/FusedBatchNormV3hu  ?B
?
?_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi256ELi64ELi16EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi256ELi16EEENS_10tfloat32_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi256ELi16EEELi128ENSG_ILi8ELi4EEELi4EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi32ELi32EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi16ELi64EEESC_NSE_INSG_ILi64ELi16EEELi128ESI_Li4EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi32ELi32EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi16EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi8EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi64ELi8ELi4ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE? ??*?2(8??<@??<H??<XbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterh
}
(ampere_scudnn_128x64_relu_interior_nn_v1???*?2?8??;@??;H??;Xb)model_1/model/block3a_project_conv/Conv2DhuMUB
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??;@??;H??;Xbmodel_1/model/top_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??;@??;H??;Xb'replica_1/model_1/model/top_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??:@??:H??:b&model_1/model/block4a_dwconv/depthwisehu  ?B
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHW<float, 3, 3, 1>(tensorflow::DepthwiseArgs, float const*, float const*, float*, int)&*?208??:@??:H??:b0replica_1/model_1/model/block4a_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)(?`*?2??8ӎ:@ӎ:Hӎ:bDmodel_1/model/stem_conv/Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizerhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??9@??9H??9b#model_1/model/block3a_se_excite/mulhuZU?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)(?`*?2??8??9@??9H??9bNreplica_1/model_1/model/stem_conv/Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizerhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??9@??9H??9b-replica_1/model_1/model/block3a_se_excite/mulhuZU?B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6>(cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6::Params)? ??*?2?8??9@??9H??9Xb3replica_1/model_1/model/block3a_project_conv/Conv2Dh
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10>(cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10::Params)f ??*?2?8??8@??8H??8Xb3replica_1/model_1/model/block3a_project_conv/Conv2Dh
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??3@??3H??3b(model_1/model/block3a_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??3@??3H??3b2replica_1/model_1/model/block3a_activation/SigmoidhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??3@??3H??3b#model_1/model/block4b_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block4d_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block7c_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4f_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block4c_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block4f_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4e_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block7b_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block4e_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block7d_se_excite/mulhuZU?B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2bmodel_1/model/block2c_add/addhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block4h_se_excite/mulhuZU?B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2bmodel_1/model/block2g_add/addhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block7d_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block7c_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block5a_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4c_se_excite/mulhuZU?B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??2@??2H??2Xb)model_1/model/block7a_project_conv/Conv2Dh
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2b'replica_1/model_1/model/block2e_add/addhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block4j_se_excite/mulhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block4i_se_excite/mulhuZU?B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2bmodel_1/model/block2e_add/addhuZU?B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b#model_1/model/block4g_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4h_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4g_se_excite/mulhuZU?B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2bmodel_1/model/block2b_add/addhuZU?B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2b'replica_1/model_1/model/block2b_add/addhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4i_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4d_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4j_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block5a_se_excite/mulhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block4b_se_excite/mulhuZU?B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2b'replica_1/model_1/model/block2f_add/addhuZU?B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2b'replica_1/model_1/model/block2c_add/addhuZU?B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2bmodel_1/model/block2d_add/addhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ҿ2@ҿ2Hҿ2b)model_1/model/block3a_bn/FusedBatchNormV3hu  ?B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2bmodel_1/model/block2f_add/addhuZU?B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38ӵ2@ӵ2Hӵ2b'replica_1/model_1/model/block2g_add/addhuZU?B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2??8??2@??2H??2b-replica_1/model_1/model/block7b_se_excite/mulhuZU?B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?38??2@??2H??2b'replica_1/model_1/model/block2d_add/addhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??2@??2H??2b3replica_1/model_1/model/block3a_bn/FusedBatchNormV3hu  ?B
?
?void cutlass_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4>(cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4::Params)? ??*?2?8??2@??2H??2Xb3replica_1/model_1/model/block7a_project_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, 3, 3, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)6 ?:*
208??.@??.H??.b&model_1/model/block7a_dwconv/depthwisehu??EB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb(model_1/model/block4b_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb(model_1/model/block4e_expand_conv/Conv2Dh
?
?void tensorflow::DepthwiseConv2dGPUKernelNCHWSmall<float, (tensorflow::DepthwiseConv2dDirection)0, 3, 3, 4, false>(tensorflow::DepthwiseArgs, float const*, float const*, float*)6 ?:*
208??.@??.H??.b0replica_1/model_1/model/block7a_dwconv/depthwisehu??EB
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb(model_1/model/block4c_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb2replica_1/model_1/model/block4g_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb(model_1/model/block4f_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb(model_1/model/block4d_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb2replica_1/model_1/model/block4f_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb(model_1/model/block4i_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb(model_1/model/block4j_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8З.@З.HЗ.PXb(model_1/model/block4g_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8Ж.@Ж.HЖ.PXb(model_1/model/block5a_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8Ђ.@Ђ.HЂ.PXb(model_1/model/block4h_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb2replica_1/model_1/model/block4h_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??.@??.H??.PXb2replica_1/model_1/model/block4i_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb2replica_1/model_1/model/block4d_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb2replica_1/model_1/model/block5a_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb2replica_1/model_1/model/block4b_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb2replica_1/model_1/model/block4e_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb2replica_1/model_1/model/block4c_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb2replica_1/model_1/model/block4j_expand_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb)model_1/model/block5b_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb)model_1/model/block5i_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb3replica_1/model_1/model/block5c_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb)model_1/model/block5d_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb)model_1/model/block5c_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb3replica_1/model_1/model/block5b_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8ў-@ў-Hў-PXb3replica_1/model_1/model/block5i_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb3replica_1/model_1/model/block5j_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8ϝ-@ϝ-Hϝ-PXb)model_1/model/block5f_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb3replica_1/model_1/model/block5d_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8ϙ-@ϙ-Hϙ-PXb)model_1/model/block5e_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8ј-@ј-Hј-PXb3replica_1/model_1/model/block5h_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb)model_1/model/block5j_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8э-@э-Hэ-PXb3replica_1/model_1/model/block5e_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8Ћ-@Ћ-HЋ-PXb)model_1/model/block5g_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??-@??-H??-PXb)model_1/model/block5h_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??,@??,H??,PXb3replica_1/model_1/model/block5f_project_conv/Conv2Dh
?
?sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s1_kernel? ??*?2?8??,@??,H??,PXb3replica_1/model_1/model/block5g_project_conv/Conv2Dh
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block4i_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block7d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block4f_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4g_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b/model_1/model/block4d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4i_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block4d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block4h_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b/model_1/model/block4h_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Ͼ,@Ͼ,HϾ,b(model_1/model/block4d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block5a_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8з,@з,Hз,b/model_1/model/block4i_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block4f_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block4g_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8ѱ,@ѱ,Hѱ,b2replica_1/model_1/model/block4j_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4e_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block4c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block7c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block5a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block7d_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block4g_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block7d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Ѧ,@Ѧ,HѦ,b2replica_1/model_1/model/block5a_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block4b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b/model_1/model/block7c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b/model_1/model/block4j_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block7b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block4e_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block7c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b/model_1/model/block7b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b/model_1/model/block7d_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4h_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b/model_1/model/block4c_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block4c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block7b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8ҍ,@ҍ,Hҍ,b9replica_1/model_1/model/block7b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4j_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block4j_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Ќ,@Ќ,HЌ,b/model_1/model/block5a_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8ό,@ό,Hό,b/model_1/model/block4g_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8Ћ,@Ћ,HЋ,b/model_1/model/block4b_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b/model_1/model/block4e_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b9replica_1/model_1/model/block4f_expand_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block4b_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b2replica_1/model_1/model/block7c_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??,@??,H??,b(model_1/model/block4i_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??+@??+H??+b(model_1/model/block4e_activation/SigmoidhuZU?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??+@??+H??+b2replica_1/model_1/model/block4h_activation/SigmoidhuZU?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?28??+@??H??Xb@gradient_tape/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?8??+@??+H??+b/replica_1/model_1/model/block1b_se_squeeze/Meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?2H8??+@??+H??+b/model_1/model/block4f_expand_activation/SigmoidhuZU?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?8??+@??+H??+b%model_1/model/block1b_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?8??+@??+H??+b/replica_1/model_1/model/block1d_se_squeeze/Meanhu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?28??+@??H??XbJgradient_tape/replica_1/model_1/model/top_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?8??+@??+H??+b/replica_1/model_1/model/block1c_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?8??+@??+H??+b%model_1/model/block1d_se_squeeze/Meanhu  ?B
?
?void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)(0*?2?8??+@??+H??+b%model_1/model/block1c_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2x8??+@??+H??+Xb)model_1/model/block7c_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4f_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b3replica_1/model_1/model/block4g_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4e_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4h_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.8??+@??+H??+Xb)model_1/model/block5a_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block4b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2x8??+@??+H??+Xb3replica_1/model_1/model/block7c_project_conv/Conv2Dhu  ?B
W
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?,8ϧ+@ϧ+Hϧ+bAdam/gradients/mulhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block4g_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4j_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b3replica_1/model_1/model/block4f_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block5a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2.8??+@??+H??+Xb3replica_1/model_1/model/block5a_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block4j_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b3replica_1/model_1/model/block4j_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8Ѥ+@Ѥ+HѤ+b:replica_1/model_1/model/block4b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8Ѥ+@Ѥ+HѤ+b3replica_1/model_1/model/block4i_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8Ϥ+@Ϥ+HϤ+b0model_1/model/block4e_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4g_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ѣ+@ѣ+Hѣ+b:replica_1/model_1/model/block4g_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ϣ+@ϣ+Hϣ+b)model_1/model/block5a_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b3replica_1/model_1/model/block4e_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block4f_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2x8??+@??+H??+Xb)model_1/model/block7d_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b3replica_1/model_1/model/block4b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block4e_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block4f_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block4c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4i_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block4d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block4d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block5a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ѡ+@ѡ+Hѡ+b3replica_1/model_1/model/block4d_bn/FusedBatchNormV3hu  ?B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?,8??+@??+H??+breplica_1/Adam/gradients/mulhuZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8Ѡ+@Ѡ+HѠ+b3replica_1/model_1/model/block4h_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2x8??+@??+H??+Xb)model_1/model/block7b_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b3replica_1/model_1/model/block5a_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ϟ+@ϟ+Hϟ+b0model_1/model/block4h_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2x8??+@??+H??+Xb3replica_1/model_1/model/block7d_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block4j_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8О+@О+HО+b:replica_1/model_1/model/block4i_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8Ξ+@Ξ+HΞ+b0model_1/model/block4i_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b3replica_1/model_1/model/block7b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block4h_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ϝ+@ϝ+Hϝ+b0model_1/model/block7d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block4c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ќ+@ќ+Hќ+b3replica_1/model_1/model/block4c_bn/FusedBatchNormV3hu  ?B
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?,8??+@??+H??+breplica_1/Adam/gradients/mul_2huZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block4d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8К+@К+HК+b3replica_1/model_1/model/block7c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)'?!*?2x8??+@??+H??+Xb3replica_1/model_1/model/block7b_project_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b3replica_1/model_1/model/block7d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block7c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ϗ+@ϗ+Hϗ+b)model_1/model/block7d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block7b_expand_bn/FusedBatchNormV3hu  ?B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?,8??+@??+H??+b*replica_1/model_1/model/top_activation/mulhuZU?B
Y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?,8??+@??+H??+bAdam/gradients/mul_1huZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b0model_1/model/block7c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block7b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b:replica_1/model_1/model/block7d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8??+@??+H??+b)model_1/model/block7b_bn/FusedBatchNormV3hu  ?B
Y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?,8ϒ+@ϒ+Hϒ+bAdam/gradients/mul_2huZU?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NCHW<float, float, true, 1>(float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float)*?2?8ѐ+@ѐ+Hѐ+b:replica_1/model_1/model/block7c_expand_bn/FusedBatchNormV3hu  ?B
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?,8??+@??+H??+breplica_1/Adam/gradients/mul_1huZU?B
?
?void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)&?!*?2x8ϊ+@ϊ+Hϊ+Xb(model_1/model/block7b_expand_conv/Conv2Dhu  ?B
e
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?,8??+@??+H??+b model_1/model/top_activation/mulhuZU?B