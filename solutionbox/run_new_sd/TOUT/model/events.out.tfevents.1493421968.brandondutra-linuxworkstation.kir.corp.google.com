       �K"	   ��@�Abrain.Event:2�=��=     IئY	M��@�A"��

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
�
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
�
)read_batch_features/file_name_queue/inputConst*
dtype0*�
value�B�B/exout/features_train-00011-of-00014.tfrecord.gzB/exout/features_train-00013-of-00014.tfrecord.gzB/exout/features_train-00001-of-00014.tfrecord.gzB/exout/features_train-00006-of-00014.tfrecord.gzB/exout/features_train-00010-of-00014.tfrecord.gzB/exout/features_train-00008-of-00014.tfrecord.gzB/exout/features_train-00004-of-00014.tfrecord.gzB/exout/features_train-00005-of-00014.tfrecord.gzB/exout/features_train-00009-of-00014.tfrecord.gzB/exout/features_train-00012-of-00014.tfrecord.gzB/exout/features_train-00000-of-00014.tfrecord.gzB/exout/features_train-00003-of-00014.tfrecord.gzB/exout/features_train-00002-of-00014.tfrecord.gzB/exout/features_train-00007-of-00014.tfrecord.gz*
_output_shapes
:
j
(read_batch_features/file_name_queue/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
o
-read_batch_features/file_name_queue/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
�
+read_batch_features/file_name_queue/GreaterGreater(read_batch_features/file_name_queue/Size-read_batch_features/file_name_queue/Greater/y*
T0*
_output_shapes
: 
�
0read_batch_features/file_name_queue/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
�
8read_batch_features/file_name_queue/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
�
1read_batch_features/file_name_queue/Assert/AssertAssert+read_batch_features/file_name_queue/Greater8read_batch_features/file_name_queue/Assert/Assert/data_0*
	summarize*

T
2
�
,read_batch_features/file_name_queue/IdentityIdentity)read_batch_features/file_name_queue/input2^read_batch_features/file_name_queue/Assert/Assert*
T0*
_output_shapes
:
�
1read_batch_features/file_name_queue/RandomShuffleRandomShuffle,read_batch_features/file_name_queue/Identity*
seed2 *

seed *
T0*
_output_shapes
:
�
#read_batch_features/file_name_queueFIFOQueueV2*
capacity *
_output_shapes
: *
shapes
: *
component_types
2*
	container *
shared_name 
�
?read_batch_features/file_name_queue/file_name_queue_EnqueueManyQueueEnqueueManyV2#read_batch_features/file_name_queue1read_batch_features/file_name_queue/RandomShuffle*

timeout_ms���������*
Tcomponents
2
�
9read_batch_features/file_name_queue/file_name_queue_CloseQueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues( 
�
;read_batch_features/file_name_queue/file_name_queue_Close_1QueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues(
�
8read_batch_features/file_name_queue/file_name_queue_SizeQueueSizeV2#read_batch_features/file_name_queue*
_output_shapes
: 
�
(read_batch_features/file_name_queue/CastCast8read_batch_features/file_name_queue/file_name_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
n
)read_batch_features/file_name_queue/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
�
'read_batch_features/file_name_queue/mulMul(read_batch_features/file_name_queue/Cast)read_batch_features/file_name_queue/mul/y*
T0*
_output_shapes
: 
�
<read_batch_features/file_name_queue/fraction_of_32_full/tagsConst*
dtype0*H
value?B= B7read_batch_features/file_name_queue/fraction_of_32_full*
_output_shapes
: 
�
7read_batch_features/file_name_queue/fraction_of_32_fullScalarSummary<read_batch_features/file_name_queue/fraction_of_32_full/tags'read_batch_features/file_name_queue/mul*
T0*
_output_shapes
: 
�
)read_batch_features/read/TFRecordReaderV2TFRecordReaderV2*
	container *
shared_name *
compression_typeGZIP*
_output_shapes
: 
w
5read_batch_features/read/ReaderReadUpToV2/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
)read_batch_features/read/ReaderReadUpToV2ReaderReadUpToV2)read_batch_features/read/TFRecordReaderV2#read_batch_features/file_name_queue5read_batch_features/read/ReaderReadUpToV2/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_1TFRecordReaderV2*
	container *
shared_name *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_1/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_1ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_1#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_1/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_2TFRecordReaderV2*
	container *
shared_name *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_2/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_2ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_2#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_2/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_3TFRecordReaderV2*
	container *
shared_name *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_3/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_3ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_3#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_3/num_records*2
_output_shapes 
:���������:���������
[
read_batch_features/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
(read_batch_features/random_shuffle_queueRandomShuffleQueueV2*
capacity*
component_types
2*
_output_shapes
: *
min_after_dequeue
*
shapes
: : *
seed2 *

seed *
	container *
shared_name 
�
read_batch_features/cond/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
q
!read_batch_features/cond/switch_tIdentity!read_batch_features/cond/Switch:1*
T0
*
_output_shapes
: 
o
!read_batch_features/cond/switch_fIdentityread_batch_features/cond/Switch*
T0
*
_output_shapes
: 
h
 read_batch_features/cond/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
@read_batch_features/cond/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue read_batch_features/cond/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1Switch)read_batch_features/read/ReaderReadUpToV2 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:���������:���������
�
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2Switch+read_batch_features/read/ReaderReadUpToV2:1 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:���������:���������
�
9read_batch_features/cond/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
+read_batch_features/cond/control_dependencyIdentity!read_batch_features/cond/switch_t:^read_batch_features/cond/random_shuffle_queue_EnqueueMany*4
_class*
(&loc:@read_batch_features/cond/switch_t*
T0
*
_output_shapes
: 
I
read_batch_features/cond/NoOpNoOp"^read_batch_features/cond/switch_f
�
-read_batch_features/cond/control_dependency_1Identity!read_batch_features/cond/switch_f^read_batch_features/cond/NoOp*4
_class*
(&loc:@read_batch_features/cond/switch_f*
T0
*
_output_shapes
: 
�
read_batch_features/cond/MergeMerge-read_batch_features/cond/control_dependency_1+read_batch_features/cond/control_dependency*
_output_shapes
: : *
T0
*
N
�
!read_batch_features/cond_1/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_1/switch_tIdentity#read_batch_features/cond_1/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_1/switch_fIdentity!read_batch_features/cond_1/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_1/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_1/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_1:1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_1/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_1/control_dependencyIdentity#read_batch_features/cond_1/switch_t<^read_batch_features/cond_1/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_1/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_1/NoOpNoOp$^read_batch_features/cond_1/switch_f
�
/read_batch_features/cond_1/control_dependency_1Identity#read_batch_features/cond_1/switch_f ^read_batch_features/cond_1/NoOp*6
_class,
*(loc:@read_batch_features/cond_1/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_1/MergeMerge/read_batch_features/cond_1/control_dependency_1-read_batch_features/cond_1/control_dependency*
_output_shapes
: : *
T0
*
N
�
!read_batch_features/cond_2/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_2/switch_tIdentity#read_batch_features/cond_2/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_2/switch_fIdentity!read_batch_features/cond_2/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_2/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_2/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_2"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_2:1"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_2/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_2/control_dependencyIdentity#read_batch_features/cond_2/switch_t<^read_batch_features/cond_2/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_2/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_2/NoOpNoOp$^read_batch_features/cond_2/switch_f
�
/read_batch_features/cond_2/control_dependency_1Identity#read_batch_features/cond_2/switch_f ^read_batch_features/cond_2/NoOp*6
_class,
*(loc:@read_batch_features/cond_2/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_2/MergeMerge/read_batch_features/cond_2/control_dependency_1-read_batch_features/cond_2/control_dependency*
_output_shapes
: : *
T0
*
N
�
!read_batch_features/cond_3/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_3/switch_tIdentity#read_batch_features/cond_3/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_3/switch_fIdentity!read_batch_features/cond_3/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_3/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_3/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_3"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_3:1"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_3/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_3/control_dependencyIdentity#read_batch_features/cond_3/switch_t<^read_batch_features/cond_3/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_3/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_3/NoOpNoOp$^read_batch_features/cond_3/switch_f
�
/read_batch_features/cond_3/control_dependency_1Identity#read_batch_features/cond_3/switch_f ^read_batch_features/cond_3/NoOp*6
_class,
*(loc:@read_batch_features/cond_3/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_3/MergeMerge/read_batch_features/cond_3/control_dependency_1-read_batch_features/cond_3/control_dependency*
_output_shapes
: : *
T0
*
N
�
.read_batch_features/random_shuffle_queue_CloseQueueCloseV2(read_batch_features/random_shuffle_queue*
cancel_pending_enqueues( 
�
0read_batch_features/random_shuffle_queue_Close_1QueueCloseV2(read_batch_features/random_shuffle_queue*
cancel_pending_enqueues(
~
-read_batch_features/random_shuffle_queue_SizeQueueSizeV2(read_batch_features/random_shuffle_queue*
_output_shapes
: 
[
read_batch_features/sub/yConst*
dtype0*
value	B :
*
_output_shapes
: 
�
read_batch_features/subSub-read_batch_features/random_shuffle_queue_Sizeread_batch_features/sub/y*
T0*
_output_shapes
: 
_
read_batch_features/Maximum/xConst*
dtype0*
value	B : *
_output_shapes
: 

read_batch_features/MaximumMaximumread_batch_features/Maximum/xread_batch_features/sub*
T0*
_output_shapes
: 
m
read_batch_features/CastCastread_batch_features/Maximum*

DstT0*

SrcT0*
_output_shapes
: 
^
read_batch_features/mul/yConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
t
read_batch_features/mulMulread_batch_features/Castread_batch_features/mul/y*
T0*
_output_shapes
: 
�
4read_batch_features/fraction_over_10_of_10_full/tagsConst*
dtype0*@
value7B5 B/read_batch_features/fraction_over_10_of_10_full*
_output_shapes
: 
�
/read_batch_features/fraction_over_10_of_10_fullScalarSummary4read_batch_features/fraction_over_10_of_10_full/tagsread_batch_features/mul*
T0*
_output_shapes
: 
W
read_batch_features/nConst*
dtype0*
value	B :
*
_output_shapes
: 
�
read_batch_featuresQueueDequeueManyV2(read_batch_features/random_shuffle_queueread_batch_features/n*

timeout_ms���������*
component_types
2* 
_output_shapes
:
:

j
(read_batch_features/ParseExample/key_keyConst*
dtype0	*
value	B	 R *
_output_shapes
: 
q
.read_batch_features/ParseExample/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
(read_batch_features/ParseExample/ReshapeReshape(read_batch_features/ParseExample/key_key.read_batch_features/ParseExample/Reshape/shape*
_output_shapes
: *
T0	*
Tshape0
i
&read_batch_features/ParseExample/ConstConst*
dtype0	*
valueB	 *
_output_shapes
: 
v
3read_batch_features/ParseExample/ParseExample/namesConst*
dtype0*
valueB *
_output_shapes
: 
�
;read_batch_features/ParseExample/ParseExample/sparse_keys_0Const*
dtype0*
valueB Btext_ids*
_output_shapes
: 
�
;read_batch_features/ParseExample/ParseExample/sparse_keys_1Const*
dtype0*
valueB Btext_weights*
_output_shapes
: 
~
:read_batch_features/ParseExample/ParseExample/dense_keys_0Const*
dtype0*
valueB	 Bkey*
_output_shapes
: 
�
:read_batch_features/ParseExample/ParseExample/dense_keys_1Const*
dtype0*
valueB Btarget*
_output_shapes
: 
�
-read_batch_features/ParseExample/ParseExampleParseExampleread_batch_features:13read_batch_features/ParseExample/ParseExample/names;read_batch_features/ParseExample/ParseExample/sparse_keys_0;read_batch_features/ParseExample/ParseExample/sparse_keys_1:read_batch_features/ParseExample/ParseExample/dense_keys_0:read_batch_features/ParseExample/ParseExample/dense_keys_1(read_batch_features/ParseExample/Reshape&read_batch_features/ParseExample/Const*
dense_shapes
: : *p
_output_shapes^
\:���������:���������:���������:���������:::
:
*
Ndense*
sparse_types
2	*
Tdense
2		*
Nsparse
�
read_batch_features/fifo_queueFIFOQueueV2*
capacityd*
_output_shapes
: *
shapes
 * 
component_types
2								*
	container *
shared_name 
j
#read_batch_features/fifo_queue_SizeQueueSizeV2read_batch_features/fifo_queue*
_output_shapes
: 
w
read_batch_features/Cast_1Cast#read_batch_features/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
`
read_batch_features/mul_1/yConst*
dtype0*
valueB
 *
�#<*
_output_shapes
: 
z
read_batch_features/mul_1Mulread_batch_features/Cast_1read_batch_features/mul_1/y*
T0*
_output_shapes
: 
�
bread_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full/tagsConst*
dtype0*n
valueeBc B]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full*
_output_shapes
: 
�
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_fullScalarSummarybread_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full/tagsread_batch_features/mul_1*
T0*
_output_shapes
: 
�
&read_batch_features/fifo_queue_enqueueQueueEnqueueV2read_batch_features/fifo_queue/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms���������*
Tcomponents
2								
�
(read_batch_features/fifo_queue_enqueue_1QueueEnqueueV2read_batch_features/fifo_queue/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms���������*
Tcomponents
2								
s
$read_batch_features/fifo_queue_CloseQueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues( 
u
&read_batch_features/fifo_queue_Close_1QueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues(
�
&read_batch_features/fifo_queue_DequeueQueueDequeueV2read_batch_features/fifo_queue*

timeout_ms���������* 
component_types
2								*v
_output_shapesd
b:
:
:���������:���������::���������:���������::

Y
ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�

ExpandDims
ExpandDims&read_batch_features/fifo_queue_DequeueExpandDims/dim*

Tdim0*
T0	*
_output_shapes

:

[
ExpandDims_1/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
ExpandDims_1
ExpandDims(read_batch_features/fifo_queue_Dequeue:1ExpandDims_1/dim*

Tdim0*
T0	*
_output_shapes

:

V
linear/linear/mod/yConst*
dtype0	*
value
B	 R�8*
_output_shapes
: 
�
linear/linear/modFloorMod(read_batch_features/fifo_queue_Dequeue:3linear/linear/mod/y*
T0	*#
_output_shapes
:���������
�
Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/ConstConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB	�8*    *
_output_shapes
:	�8
�
7linear/text_ids_weighted_by_text_weights/weights/part_0
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign7linear/text_ids_weighted_by_text_weights/weights/part_0Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/Const*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
<linear/text_ids_weighted_by_text_weights/weights/part_0/readIdentity7linear/text_ids_weighted_by_text_weights/weights/part_0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SliceSlice(read_batch_features/fifo_queue_Dequeue:4elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/begindlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ProdProd_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
�
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GatherGather(read_batch_features/fifo_queue_Dequeue:4hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
�
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shapePack^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Prod`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather*
N*
T0	*
_output_shapes
:*

axis 
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeSparseReshape(read_batch_features/fifo_queue_Dequeue:2(read_batch_features/fifo_queue_Dequeue:4qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shape*-
_output_shapes
:���������:
�
plinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/IdentityIdentitylinear/linear/mod*
T0	*#
_output_shapes
:���������
�
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualGreaterEqualplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityhlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/y*
T0	*#
_output_shapes
:���������
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterGreater(read_batch_features/fifo_queue_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/y*
T0*#
_output_shapes
:���������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd
LogicalAndflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater*#
_output_shapes
:���������
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/WhereWheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:���������
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ReshapeReshape_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Whereglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shape*#
_output_shapes
:���������*
T0	*
Tshape0
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2Gatherplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityIdentityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1Wheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:���������
�
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1Reshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shape*#
_output_shapes
:���������*
T0	*
Tshape0
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4Gather(read_batch_features/fifo_queue_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Identityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_sliceStridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/CastCast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/rangeRangeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startrlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Castylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:���������
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1Castslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffListDifftlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims
ExpandDims}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseSparseToDensevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_values�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ReshapeReshapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shape*'
_output_shapes
:���������*
T0	*
Tshape0
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_like	ZerosLikeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:���������
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concatConcatV2ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_likeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0	
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ShapeShapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
�
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/FillFillslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Shapeslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0	
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Fill{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axis*
N*

Tidx0*#
_output_shapes
:���������*
T0	
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorderSparseReordervlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*6
_output_shapes$
":���������:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/IdentityIdentityblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*
_output_shapes
:
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_sliceStridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/CastCast}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/rangeRange{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/starttlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/delta*

Tidx0*#
_output_shapes
:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1Castulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range*

DstT0	*

SrcT0*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffListDiffvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2StridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims
ExpandDimslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDenseSparseToDensexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffzlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_values�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:���������
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ReshapeReshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shape*'
_output_shapes
:���������*
T0	*
Tshape0
�
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like	ZerosLikewlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape*
T0	*'
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshapezlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0	
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ShapeShapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff*
out_type0*
T0	*
_output_shapes
:
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/FillFillulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Shapeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Const*
T0*#
_output_shapes
:���������
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0	
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axis*
N*

Tidx0*#
_output_shapes
:���������*
T0
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorderSparseReorderxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0*6
_output_shapes$
":���������:���������
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/IdentityIdentitydlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_sliceStridedSlice{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/CastCastlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupGather<linear/text_ids_weighted_by_text_weights/weights/part_0/read}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
validate_indices(*
Tparams0*
Tindices0	*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*'
_output_shapes
:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/RankConst*
dtype0*
value	B :*
_output_shapes
: 
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/subSubvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Rankwlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/y*
T0*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims
ExpandDimsulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/FillFill|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/value*
T0*#
_output_shapes
:���������
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ShapeShapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1*
out_type0*
T0*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Shapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axis*
N*

Tidx0*#
_output_shapes
:���������*
T0
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ReshapeReshapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat*'
_output_shapes
:���������*
T0*
Tshape0
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulMul�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:���������
�
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse
SegmentSumulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:���������
�
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2Reshape{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shape*'
_output_shapes
:���������*
T0
*
Tshape0
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ShapeShapeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
�
mlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_sliceStridedSlice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shapemlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackolinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stackPackalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/TileTileclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:������������������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like	ZerosLikeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
Ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weightsSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tiledlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_likeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/CastCast(read_batch_features/fifo_queue_Dequeue:4*

DstT0*

SrcT0	*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1Slice^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Castglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1ShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/sizeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2Slicealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concatConcatV2alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3ReshapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat*'
_output_shapes
:���������*
T0*
Tshape0
l
linear/linear/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
linear/linear/ReshapeReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3linear/linear/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
+linear/bias_weight/part_0/Initializer/ConstConst*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
valueB*    *
_output_shapes
:
�
linear/bias_weight/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
 linear/bias_weight/part_0/AssignAssignlinear/bias_weight/part_0+linear/bias_weight/part_0/Initializer/Const*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
linear/bias_weight/part_0/readIdentitylinear/bias_weight/part_0*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
c
linear/bias_weightIdentitylinear/bias_weight/part_0/read*
T0*
_output_shapes
:
�
linear/linear/BiasAddBiasAddlinear/linear/Reshapelinear/bias_weight*'
_output_shapes
:���������*
T0*
data_formatNHWC
m
predictions/probabilitiesSoftmaxlinear/linear/BiasAdd*
T0*'
_output_shapes
:���������
_
predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
predictions/classesArgMaxlinear/linear/BiasAddpredictions/classes/dimension*#
_output_shapes
:���������*
T0*

Tidx0
�
0training_loss/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*
_output_shapes
:

x
.training_loss/softmax_cross_entropy_loss/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
�
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*$
_output_shapes
:
:

]
training_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
[
train_op/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
]
train_op/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
t
train_op/gradients/FillFilltrain_op/gradients/Shapetrain_op/gradients/Const*
T0*
_output_shapes
: 
}
3train_op/gradients/training_loss_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
-train_op/gradients/training_loss_grad/ReshapeReshapetrain_op/gradients/Fill3train_op/gradients/training_loss_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
~
4train_op/gradients/training_loss_grad/Tile/multiplesConst*
dtype0*
valueB:
*
_output_shapes
:
�
*train_op/gradients/training_loss_grad/TileTile-train_op/gradients/training_loss_grad/Reshape4train_op/gradients/training_loss_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:

u
+train_op/gradients/training_loss_grad/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
p
-train_op/gradients/training_loss_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
u
+train_op/gradients/training_loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
*train_op/gradients/training_loss_grad/ProdProd+train_op/gradients/training_loss_grad/Shape+train_op/gradients/training_loss_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
w
-train_op/gradients/training_loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
,train_op/gradients/training_loss_grad/Prod_1Prod-train_op/gradients/training_loss_grad/Shape_1-train_op/gradients/training_loss_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
q
/train_op/gradients/training_loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
-train_op/gradients/training_loss_grad/MaximumMaximum,train_op/gradients/training_loss_grad/Prod_1/train_op/gradients/training_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
.train_op/gradients/training_loss_grad/floordivFloorDiv*train_op/gradients/training_loss_grad/Prod-train_op/gradients/training_loss_grad/Maximum*
T0*
_output_shapes
: 
�
*train_op/gradients/training_loss_grad/CastCast.train_op/gradients/training_loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
-train_op/gradients/training_loss_grad/truedivRealDiv*train_op/gradients/training_loss_grad/Tile*train_op/gradients/training_loss_grad/Cast*
T0*
_output_shapes
:


train_op/gradients/zeros_like	ZerosLike*training_loss/softmax_cross_entropy_loss:1*
T0*
_output_shapes

:

�
Ptrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/PreventGradientPreventGradient*training_loss/softmax_cross_entropy_loss:1*
T0*
_output_shapes

:

�
Otrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
Ktrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims
ExpandDims-train_op/gradients/training_loss_grad/truedivOtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:

�
Dtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mulMulKtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDimsPtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/PreventGradient*
T0*
_output_shapes

:

�
9train_op/gradients/linear/linear/BiasAdd_grad/BiasAddGradBiasAddGradDtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mul*
_output_shapes
:*
T0*
data_formatNHWC
�
3train_op/gradients/linear/linear/Reshape_grad/ShapeShapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3*
out_type0*
T0*
_output_shapes
:
�
5train_op/gradients/linear/linear/Reshape_grad/ReshapeReshapeDtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mul3train_op/gradients/linear/linear/Reshape_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/ShapeShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/ReshapeReshape5train_op/gradients/linear/linear/Reshape_grad/Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like	ZerosLikedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like*
T0*'
_output_shapes
:���������
�
xtrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/SelectSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tile�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Reshape|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like*
T0*
_output_shapes

:

�
ztrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/Select_1Select^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tile|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Reshape*
T0*
_output_shapes

:

�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/GatherGatherztrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/Select_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
validate_indices(*
Tparams0*
Tindices0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/ShapeShape�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1Shapeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgsBroadcastGradientArgs�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mulMul�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/Gatherylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/SumSum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/ReshapeReshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul_1Mul�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/Gather*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum_1Sum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Reshape_1Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ShapeConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB"     *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/SizeSize}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
out_type0*
T0	*
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims
ExpandDims�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Size�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_sliceStridedSlice�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Shape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concatConcatV2�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ReshapeReshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat*0
_output_shapes
:������������������*
T0*
Tshape0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape_1Reshape}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims*#
_output_shapes
:���������*
T0	*
Tshape0
�
"train_op/beta1_power/initial_valueConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *fff?*
_output_shapes
: 
�
train_op/beta1_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
train_op/beta1_power/AssignAssigntrain_op/beta1_power"train_op/beta1_power/initial_value*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
train_op/beta1_power/readIdentitytrain_op/beta1_power*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
"train_op/beta2_power/initial_valueConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *w�?*
_output_shapes
: 
�
train_op/beta2_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
train_op/beta2_power/AssignAssigntrain_op/beta2_power"train_op/beta2_power/initial_value*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
train_op/beta2_power/readIdentitytrain_op/beta2_power*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
e
train_op/zerosConst*
dtype0*
valueB	�8*    *
_output_shapes
:	�8
�
<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/AssignAssign<linear/text_ids_weighted_by_text_weights/weights/part_0/Adamtrain_op/zeros*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
Alinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readIdentity<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
g
train_op/zeros_1Const*
dtype0*
valueB	�8*    *
_output_shapes
:	�8
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
Elinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/AssignAssign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1train_op/zeros_1*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readIdentity>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
]
train_op/zeros_2Const*
dtype0*
valueB*    *
_output_shapes
:
�
linear/bias_weight/part_0/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
%linear/bias_weight/part_0/Adam/AssignAssignlinear/bias_weight/part_0/Adamtrain_op/zeros_2*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
#linear/bias_weight/part_0/Adam/readIdentitylinear/bias_weight/part_0/Adam*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
]
train_op/zeros_3Const*
dtype0*
valueB*    *
_output_shapes
:
�
 linear/bias_weight/part_0/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
'linear/bias_weight/part_0/Adam_1/AssignAssign linear/bias_weight/part_0/Adam_1train_op/zeros_3*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
%linear/bias_weight/part_0/Adam_1/readIdentity linear/bias_weight/part_0/Adam_1*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
`
train_op/Adam/learning_rateConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
X
train_op/Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
X
train_op/Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Z
train_op/Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueUnique�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ShapeShapeStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Unique*
out_type0*
T0	*
_output_shapes
:*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
`train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stackConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB: *
_output_shapes
:
�
btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_1Const*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB:*
_output_shapes
:
�
btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_2Const*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB:*
_output_shapes
:
�
Ztrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_sliceStridedSliceRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Shape`train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stackbtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_1btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSumUnsortedSegmentSum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ReshapeUtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Unique:1Ztrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice*
Tindices0*
T0*0
_output_shapes
:������������������*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/subSubRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub/xtrain_op/beta2_power/read*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Qtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/SqrtSqrtPtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mulMultrain_op/Adam/learning_rateQtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1/xtrain_op/beta1_power/read*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truedivRealDivPtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mulRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2/xtrain_op/Adam/beta1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_1Mul_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSumRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_2MulAlinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readtrain_op/Adam/beta1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign<linear/text_ids_weighted_by_text_weights/weights/part_0/AdamRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_2*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Wtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd
ScatterAddStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_1*
Tindices0	*
use_locking( *
T0*
_output_shapes
:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_3Mul_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSum_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSum*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3/xtrain_op/Adam/beta2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_4MulRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_3Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_5MulClinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readtrain_op/Adam/beta2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Utrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Assign_1Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_5*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Ytrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1
ScatterAddUtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Assign_1Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_4*
Tindices0	*
use_locking( *
T0*
_output_shapes
:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt_1SqrtYtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_6MulTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truedivWtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/addAddStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt_1train_op/Adam/epsilon*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truediv_1RealDivRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_6Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/add*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignSub	AssignSub7linear/text_ids_weighted_by_text_weights/weights/part_0Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truediv_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Wtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_depsNoOpW^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignSubX^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAddZ^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
8train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam	ApplyAdamlinear/bias_weight/part_0linear/bias_weight/part_0/Adam linear/bias_weight/part_0/Adam_1train_op/beta1_power/readtrain_op/beta2_power/readtrain_op/Adam/learning_ratetrain_op/Adam/beta1train_op/Adam/beta2train_op/Adam/epsilon9train_op/gradients/linear/linear/BiasAdd_grad/BiasAddGrad*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking( *
T0*
_output_shapes
:
�
train_op/Adam/mulMultrain_op/beta1_power/readtrain_op/Adam/beta1X^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
train_op/Adam/AssignAssigntrain_op/beta1_powertrain_op/Adam/mul*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
: 
�
train_op/Adam/mul_1Multrain_op/beta2_power/readtrain_op/Adam/beta2X^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
train_op/Adam/Assign_1Assigntrain_op/beta2_powertrain_op/Adam/mul_1*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
: 
�
train_op/Adam/updateNoOpX^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam^train_op/Adam/Assign^train_op/Adam/Assign_1
�
train_op/Adam/valueConst^train_op/Adam/update*
dtype0	*
_class
loc:@global_step*
value	B	 R*
_output_shapes
: 
�
train_op/Adam	AssignAddglobal_steptrain_op/Adam/value*
_class
loc:@global_step*
use_locking( *
T0	*
_output_shapes
: 
�
,metrics/remove_squeezable_dimensions/SqueezeSqueezeExpandDims_1*
squeeze_dims

���������*
T0	*
_output_shapes
:

~
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*
_output_shapes
:

Z
metrics/ToFloatCastmetrics/Equal*

DstT0*

SrcT0
*
_output_shapes
:

[
metrics/accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
validate_shape(*)
_class
loc:@metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
�
metrics/accuracy/total/readIdentitymetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
T0*
_output_shapes
: 
]
metrics/accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
validate_shape(*)
_class
loc:@metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
�
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
T0*
_output_shapes
: 
W
metrics/accuracy/SizeConst*
dtype0*
value	B :
*
_output_shapes
: 
i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
`
metrics/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*)
_class
loc:@metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/accuracy/AssignAdd_1	AssignAddmetrics/accuracy/countmetrics/accuracy/ToFloat_1*)
_class
loc:@metrics/accuracy/count*
use_locking( *
T0*
_output_shapes
: 
_
metrics/accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
}
metrics/accuracy/GreaterGreatermetrics/accuracy/count/readmetrics/accuracy/Greater/y*
T0*
_output_shapes
: 
~
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
T0*
_output_shapes
: 
]
metrics/accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
T0*
_output_shapes
: 
a
metrics/accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
�
metrics/accuracy/truediv_1RealDivmetrics/accuracy/AssignAddmetrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
a
metrics/accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
N
metrics/RankConst*
dtype0*
value	B :*
_output_shapes
: 
U
metrics/LessEqual/yConst*
dtype0*
value	B :*
_output_shapes
: 
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
T0*
_output_shapes
: 
�
metrics/Assert/ConstConst*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
�
metrics/Assert/Assert/data_0Const*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*
	summarize*

T
2
�
metrics/Reshape/shapeConst^metrics/Assert/Assert*
dtype0*
valueB:
���������*
_output_shapes
:
r
metrics/ReshapeReshapeExpandDims_1metrics/Reshape/shape*
_output_shapes
:
*
T0	*
Tshape0
]
metrics/one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
^
metrics/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
W
metrics/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
axis���������*
T0*
_output_shapes

:
*
TI0	
]
metrics/CastCastmetrics/one_hot*

DstT0
*

SrcT0*
_output_shapes

:

j
metrics/auc/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"   ����*
_output_shapes
:
�
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
_output_shapes
:	�*
T0
*
Tshape0
d
metrics/auc/ShapeShapemetrics/auc/Reshape*
out_type0*
T0*
_output_shapes
:
i
metrics/auc/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
k
!metrics/auc/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
k
!metrics/auc/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_sliceStridedSlicemetrics/auc/Shapemetrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
�
metrics/auc/ConstConst*
dtype0*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
_output_shapes	
:�
d
metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	�
U
metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
N*
T0*
_output_shapes
:*

axis 
�
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:����������
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
T0*
_output_shapes
: 
]
metrics/auc/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
T0*
_output_shapes
:
�
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:���������
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"�      *
_output_shapes
:
�
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
w
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*(
_output_shapes
:����������
c
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*(
_output_shapes
:����������
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"�      *
_output_shapes
:
�
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
* 
_output_shapes
:
��
\
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2* 
_output_shapes
:
��
`
metrics/auc/zerosConst*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/true_positives
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:�
�
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
T0*
_output_shapes	
:�
o
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater* 
_output_shapes
:
��
o
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

DstT0*

SrcT0
* 
_output_shapes
:
��
c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*-
_class#
!loc:@metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_1Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/false_negatives
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
validate_shape(*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:�
�
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
_output_shapes	
:�
t
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot* 
_output_shapes
:
��
q
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_2Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/true_negatives
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:�
�
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
_output_shapes	
:�
z
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot* 
_output_shapes
:
��
q
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_3Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/false_positives
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:�
�
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
T0*
_output_shapes	
:�
w
metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater* 
_output_shapes
:
��
q
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*.
_class$
" loc:@metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:�
V
metrics/auc/add/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:�
�
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_2/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
T0*
_output_shapes	
:�
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
T0*
_output_shapes	
:�
�
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_4/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:�
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:�
Z
metrics/auc/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
T0*
_output_shapes	
:�
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:�
]
metrics/auc/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
|
metrics/auc/valueSummetrics/auc/Mulmetrics/auc/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
X
metrics/auc/add_6/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
T0*
_output_shapes	
:�
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:�
X
metrics/auc/add_8/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:�
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
T0*
_output_shapes	
:�
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:�
Y
metrics/auc/add_10/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:�
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
T0*
_output_shapes	
:�
\
metrics/auc/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
s
metrics/auc/truediv_1RealDivmetrics/auc/add_11metrics/auc/truediv_1/y*
T0*
_output_shapes	
:�
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:�
]
metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
�
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

*metrics/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*
_output_shapes
:

r
(metrics/softmax_cross_entropy_loss/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
�
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*$
_output_shapes
:
:

a
metrics/eval_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: "���X     ���
	�`%��@�AJ�	
�;�;
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
p
	AssignSub
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint���������"
	containerstring "
shared_namestring �
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
�
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
<
	LessEqual
x"T
y"T
z
"
Ttype:
2		
\
ListDiff
x"T
y"T
out"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
$

LogicalAnd
x

y

z
�


LogicalNot
x

y

:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParseExample

serialized	
names
sparse_keys*Nsparse

dense_keys*Ndense
dense_defaults2Tdense
sparse_indices	*Nsparse
sparse_values2sparse_types
sparse_shapes	*Nsparse
dense_values2Tdense"
Nsparseint("
Ndenseint("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
5
PreventGradient

input"T
output"T"	
Ttype
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
?
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( 
�
QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint���������
~
QueueDequeueV2

handle

components2component_types"!
component_types
list(type)(0"

timeout_msint���������
z
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint���������
v
QueueEnqueueV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint���������
#
QueueSizeV2

handle
size
Y
RandomShuffle

value"T
output"T"
seedint "
seed2int "	
Ttype�
�
RandomShuffleQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint���������"
min_after_dequeueint "
seedint "
seed2int "
	containerstring "
shared_namestring �
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
^
ReaderReadUpToV2
reader_handle
queue_handle
num_records	
keys

values
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
�

ScatterAdd
ref"T�
indices"Tindices
updates"T

output_ref"T�"
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
v

SegmentSum	
data"T
segment_ids"Tindices
output"T"
Ttype:
2	"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
y
SparseReorder
input_indices	
input_values"T
input_shape	
output_indices	
output_values"T"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
�
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
�
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
,
Sqrt
x"T
y"T"
Ttype:	
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
|
TFRecordReaderV2
reader_handle"
	containerstring "
shared_namestring "
compression_typestring �
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
�
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments
output"T"
Ttype:
2	"
Tindicestype:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirty��

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
�
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
�
)read_batch_features/file_name_queue/inputConst*
dtype0*�
value�B�B/exout/features_train-00011-of-00014.tfrecord.gzB/exout/features_train-00013-of-00014.tfrecord.gzB/exout/features_train-00001-of-00014.tfrecord.gzB/exout/features_train-00006-of-00014.tfrecord.gzB/exout/features_train-00010-of-00014.tfrecord.gzB/exout/features_train-00008-of-00014.tfrecord.gzB/exout/features_train-00004-of-00014.tfrecord.gzB/exout/features_train-00005-of-00014.tfrecord.gzB/exout/features_train-00009-of-00014.tfrecord.gzB/exout/features_train-00012-of-00014.tfrecord.gzB/exout/features_train-00000-of-00014.tfrecord.gzB/exout/features_train-00003-of-00014.tfrecord.gzB/exout/features_train-00002-of-00014.tfrecord.gzB/exout/features_train-00007-of-00014.tfrecord.gz*
_output_shapes
:
j
(read_batch_features/file_name_queue/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
o
-read_batch_features/file_name_queue/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
�
+read_batch_features/file_name_queue/GreaterGreater(read_batch_features/file_name_queue/Size-read_batch_features/file_name_queue/Greater/y*
T0*
_output_shapes
: 
�
0read_batch_features/file_name_queue/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
�
8read_batch_features/file_name_queue/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
�
1read_batch_features/file_name_queue/Assert/AssertAssert+read_batch_features/file_name_queue/Greater8read_batch_features/file_name_queue/Assert/Assert/data_0*
	summarize*

T
2
�
,read_batch_features/file_name_queue/IdentityIdentity)read_batch_features/file_name_queue/input2^read_batch_features/file_name_queue/Assert/Assert*
T0*
_output_shapes
:
�
1read_batch_features/file_name_queue/RandomShuffleRandomShuffle,read_batch_features/file_name_queue/Identity*
seed2 *

seed *
T0*
_output_shapes
:
�
#read_batch_features/file_name_queueFIFOQueueV2*
capacity *
component_types
2*
_output_shapes
: *
shapes
: *
	container *
shared_name 
�
?read_batch_features/file_name_queue/file_name_queue_EnqueueManyQueueEnqueueManyV2#read_batch_features/file_name_queue1read_batch_features/file_name_queue/RandomShuffle*

timeout_ms���������*
Tcomponents
2
�
9read_batch_features/file_name_queue/file_name_queue_CloseQueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues( 
�
;read_batch_features/file_name_queue/file_name_queue_Close_1QueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues(
�
8read_batch_features/file_name_queue/file_name_queue_SizeQueueSizeV2#read_batch_features/file_name_queue*
_output_shapes
: 
�
(read_batch_features/file_name_queue/CastCast8read_batch_features/file_name_queue/file_name_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
n
)read_batch_features/file_name_queue/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
�
'read_batch_features/file_name_queue/mulMul(read_batch_features/file_name_queue/Cast)read_batch_features/file_name_queue/mul/y*
T0*
_output_shapes
: 
�
<read_batch_features/file_name_queue/fraction_of_32_full/tagsConst*
dtype0*H
value?B= B7read_batch_features/file_name_queue/fraction_of_32_full*
_output_shapes
: 
�
7read_batch_features/file_name_queue/fraction_of_32_fullScalarSummary<read_batch_features/file_name_queue/fraction_of_32_full/tags'read_batch_features/file_name_queue/mul*
T0*
_output_shapes
: 
�
)read_batch_features/read/TFRecordReaderV2TFRecordReaderV2*
shared_name *
	container *
compression_typeGZIP*
_output_shapes
: 
w
5read_batch_features/read/ReaderReadUpToV2/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
)read_batch_features/read/ReaderReadUpToV2ReaderReadUpToV2)read_batch_features/read/TFRecordReaderV2#read_batch_features/file_name_queue5read_batch_features/read/ReaderReadUpToV2/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_1TFRecordReaderV2*
shared_name *
	container *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_1/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_1ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_1#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_1/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_2TFRecordReaderV2*
shared_name *
	container *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_2/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_2ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_2#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_2/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_3TFRecordReaderV2*
shared_name *
	container *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_3/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_3ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_3#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_3/num_records*2
_output_shapes 
:���������:���������
[
read_batch_features/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
(read_batch_features/random_shuffle_queueRandomShuffleQueueV2*
	container *
component_types
2*
_output_shapes
: *
min_after_dequeue
*
shapes
: : *
seed2 *

seed *
capacity*
shared_name 
�
read_batch_features/cond/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
q
!read_batch_features/cond/switch_tIdentity!read_batch_features/cond/Switch:1*
T0
*
_output_shapes
: 
o
!read_batch_features/cond/switch_fIdentityread_batch_features/cond/Switch*
T0
*
_output_shapes
: 
h
 read_batch_features/cond/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
@read_batch_features/cond/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue read_batch_features/cond/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1Switch)read_batch_features/read/ReaderReadUpToV2 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:���������:���������
�
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2Switch+read_batch_features/read/ReaderReadUpToV2:1 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:���������:���������
�
9read_batch_features/cond/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
+read_batch_features/cond/control_dependencyIdentity!read_batch_features/cond/switch_t:^read_batch_features/cond/random_shuffle_queue_EnqueueMany*4
_class*
(&loc:@read_batch_features/cond/switch_t*
T0
*
_output_shapes
: 
I
read_batch_features/cond/NoOpNoOp"^read_batch_features/cond/switch_f
�
-read_batch_features/cond/control_dependency_1Identity!read_batch_features/cond/switch_f^read_batch_features/cond/NoOp*4
_class*
(&loc:@read_batch_features/cond/switch_f*
T0
*
_output_shapes
: 
�
read_batch_features/cond/MergeMerge-read_batch_features/cond/control_dependency_1+read_batch_features/cond/control_dependency*
N*
T0
*
_output_shapes
: : 
�
!read_batch_features/cond_1/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_1/switch_tIdentity#read_batch_features/cond_1/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_1/switch_fIdentity!read_batch_features/cond_1/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_1/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_1/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_1:1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_1/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_1/control_dependencyIdentity#read_batch_features/cond_1/switch_t<^read_batch_features/cond_1/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_1/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_1/NoOpNoOp$^read_batch_features/cond_1/switch_f
�
/read_batch_features/cond_1/control_dependency_1Identity#read_batch_features/cond_1/switch_f ^read_batch_features/cond_1/NoOp*6
_class,
*(loc:@read_batch_features/cond_1/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_1/MergeMerge/read_batch_features/cond_1/control_dependency_1-read_batch_features/cond_1/control_dependency*
N*
T0
*
_output_shapes
: : 
�
!read_batch_features/cond_2/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_2/switch_tIdentity#read_batch_features/cond_2/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_2/switch_fIdentity!read_batch_features/cond_2/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_2/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_2/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_2"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_2:1"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_2/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_2/control_dependencyIdentity#read_batch_features/cond_2/switch_t<^read_batch_features/cond_2/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_2/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_2/NoOpNoOp$^read_batch_features/cond_2/switch_f
�
/read_batch_features/cond_2/control_dependency_1Identity#read_batch_features/cond_2/switch_f ^read_batch_features/cond_2/NoOp*6
_class,
*(loc:@read_batch_features/cond_2/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_2/MergeMerge/read_batch_features/cond_2/control_dependency_1-read_batch_features/cond_2/control_dependency*
N*
T0
*
_output_shapes
: : 
�
!read_batch_features/cond_3/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_3/switch_tIdentity#read_batch_features/cond_3/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_3/switch_fIdentity!read_batch_features/cond_3/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_3/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_3/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_3"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_3:1"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_3/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_3/control_dependencyIdentity#read_batch_features/cond_3/switch_t<^read_batch_features/cond_3/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_3/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_3/NoOpNoOp$^read_batch_features/cond_3/switch_f
�
/read_batch_features/cond_3/control_dependency_1Identity#read_batch_features/cond_3/switch_f ^read_batch_features/cond_3/NoOp*6
_class,
*(loc:@read_batch_features/cond_3/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_3/MergeMerge/read_batch_features/cond_3/control_dependency_1-read_batch_features/cond_3/control_dependency*
N*
T0
*
_output_shapes
: : 
�
.read_batch_features/random_shuffle_queue_CloseQueueCloseV2(read_batch_features/random_shuffle_queue*
cancel_pending_enqueues( 
�
0read_batch_features/random_shuffle_queue_Close_1QueueCloseV2(read_batch_features/random_shuffle_queue*
cancel_pending_enqueues(
~
-read_batch_features/random_shuffle_queue_SizeQueueSizeV2(read_batch_features/random_shuffle_queue*
_output_shapes
: 
[
read_batch_features/sub/yConst*
dtype0*
value	B :
*
_output_shapes
: 
�
read_batch_features/subSub-read_batch_features/random_shuffle_queue_Sizeread_batch_features/sub/y*
T0*
_output_shapes
: 
_
read_batch_features/Maximum/xConst*
dtype0*
value	B : *
_output_shapes
: 

read_batch_features/MaximumMaximumread_batch_features/Maximum/xread_batch_features/sub*
T0*
_output_shapes
: 
m
read_batch_features/CastCastread_batch_features/Maximum*

DstT0*

SrcT0*
_output_shapes
: 
^
read_batch_features/mul/yConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
t
read_batch_features/mulMulread_batch_features/Castread_batch_features/mul/y*
T0*
_output_shapes
: 
�
4read_batch_features/fraction_over_10_of_10_full/tagsConst*
dtype0*@
value7B5 B/read_batch_features/fraction_over_10_of_10_full*
_output_shapes
: 
�
/read_batch_features/fraction_over_10_of_10_fullScalarSummary4read_batch_features/fraction_over_10_of_10_full/tagsread_batch_features/mul*
T0*
_output_shapes
: 
W
read_batch_features/nConst*
dtype0*
value	B :
*
_output_shapes
: 
�
read_batch_featuresQueueDequeueManyV2(read_batch_features/random_shuffle_queueread_batch_features/n*

timeout_ms���������*
component_types
2* 
_output_shapes
:
:

j
(read_batch_features/ParseExample/key_keyConst*
dtype0	*
value	B	 R *
_output_shapes
: 
q
.read_batch_features/ParseExample/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
(read_batch_features/ParseExample/ReshapeReshape(read_batch_features/ParseExample/key_key.read_batch_features/ParseExample/Reshape/shape*
Tshape0*
T0	*
_output_shapes
: 
i
&read_batch_features/ParseExample/ConstConst*
dtype0	*
valueB	 *
_output_shapes
: 
v
3read_batch_features/ParseExample/ParseExample/namesConst*
dtype0*
valueB *
_output_shapes
: 
�
;read_batch_features/ParseExample/ParseExample/sparse_keys_0Const*
dtype0*
valueB Btext_ids*
_output_shapes
: 
�
;read_batch_features/ParseExample/ParseExample/sparse_keys_1Const*
dtype0*
valueB Btext_weights*
_output_shapes
: 
~
:read_batch_features/ParseExample/ParseExample/dense_keys_0Const*
dtype0*
valueB	 Bkey*
_output_shapes
: 
�
:read_batch_features/ParseExample/ParseExample/dense_keys_1Const*
dtype0*
valueB Btarget*
_output_shapes
: 
�
-read_batch_features/ParseExample/ParseExampleParseExampleread_batch_features:13read_batch_features/ParseExample/ParseExample/names;read_batch_features/ParseExample/ParseExample/sparse_keys_0;read_batch_features/ParseExample/ParseExample/sparse_keys_1:read_batch_features/ParseExample/ParseExample/dense_keys_0:read_batch_features/ParseExample/ParseExample/dense_keys_1(read_batch_features/ParseExample/Reshape&read_batch_features/ParseExample/Const*
dense_shapes
: : *p
_output_shapes^
\:���������:���������:���������:���������:::
:
*
Ndense*
sparse_types
2	*
Tdense
2		*
Nsparse
�
read_batch_features/fifo_queueFIFOQueueV2*
capacityd* 
component_types
2								*
_output_shapes
: *
shapes
 *
	container *
shared_name 
j
#read_batch_features/fifo_queue_SizeQueueSizeV2read_batch_features/fifo_queue*
_output_shapes
: 
w
read_batch_features/Cast_1Cast#read_batch_features/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
`
read_batch_features/mul_1/yConst*
dtype0*
valueB
 *
�#<*
_output_shapes
: 
z
read_batch_features/mul_1Mulread_batch_features/Cast_1read_batch_features/mul_1/y*
T0*
_output_shapes
: 
�
bread_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full/tagsConst*
dtype0*n
valueeBc B]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full*
_output_shapes
: 
�
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_fullScalarSummarybread_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full/tagsread_batch_features/mul_1*
T0*
_output_shapes
: 
�
&read_batch_features/fifo_queue_enqueueQueueEnqueueV2read_batch_features/fifo_queue/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms���������*
Tcomponents
2								
�
(read_batch_features/fifo_queue_enqueue_1QueueEnqueueV2read_batch_features/fifo_queue/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms���������*
Tcomponents
2								
s
$read_batch_features/fifo_queue_CloseQueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues( 
u
&read_batch_features/fifo_queue_Close_1QueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues(
�
&read_batch_features/fifo_queue_DequeueQueueDequeueV2read_batch_features/fifo_queue*

timeout_ms���������* 
component_types
2								*v
_output_shapesd
b:
:
:���������:���������::���������:���������::

Y
ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�

ExpandDims
ExpandDims&read_batch_features/fifo_queue_DequeueExpandDims/dim*

Tdim0*
T0	*
_output_shapes

:

[
ExpandDims_1/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
ExpandDims_1
ExpandDims(read_batch_features/fifo_queue_Dequeue:1ExpandDims_1/dim*

Tdim0*
T0	*
_output_shapes

:

V
linear/linear/mod/yConst*
dtype0	*
value
B	 R�8*
_output_shapes
: 
�
linear/linear/modFloorMod(read_batch_features/fifo_queue_Dequeue:3linear/linear/mod/y*
T0	*#
_output_shapes
:���������
�
Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/ConstConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB	�8*    *
_output_shapes
:	�8
�
7linear/text_ids_weighted_by_text_weights/weights/part_0
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign7linear/text_ids_weighted_by_text_weights/weights/part_0Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/Const*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
<linear/text_ids_weighted_by_text_weights/weights/part_0/readIdentity7linear/text_ids_weighted_by_text_weights/weights/part_0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SliceSlice(read_batch_features/fifo_queue_Dequeue:4elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/begindlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ProdProd_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
�
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GatherGather(read_batch_features/fifo_queue_Dequeue:4hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
�
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shapePack^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Prod`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather*
_output_shapes
:*

axis *
T0	*
N
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeSparseReshape(read_batch_features/fifo_queue_Dequeue:2(read_batch_features/fifo_queue_Dequeue:4qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shape*-
_output_shapes
:���������:
�
plinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/IdentityIdentitylinear/linear/mod*
T0	*#
_output_shapes
:���������
�
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualGreaterEqualplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityhlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/y*
T0	*#
_output_shapes
:���������
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterGreater(read_batch_features/fifo_queue_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/y*
T0*#
_output_shapes
:���������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd
LogicalAndflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater*#
_output_shapes
:���������
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/WhereWheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:���������
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ReshapeReshape_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Whereglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2Gatherplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityIdentityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1Wheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:���������
�
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1Reshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shape*
Tshape0*
T0	*#
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4Gather(read_batch_features/fifo_queue_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Identityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_sliceStridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/CastCast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/rangeRangeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startrlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Castylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:���������
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1Castslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffListDifftlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims
ExpandDims}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseSparseToDensevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_values�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ReshapeReshapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:���������
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_like	ZerosLikeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:���������
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concatConcatV2ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_likeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axis*'
_output_shapes
:���������*

Tidx0*
T0	*
N
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ShapeShapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
�
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/FillFillslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Shapeslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:���������*

Tidx0*
T0	*
N
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Fill{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:���������*

Tidx0*
T0	*
N
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorderSparseReordervlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*6
_output_shapes$
":���������:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/IdentityIdentityblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*
_output_shapes
:
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_sliceStridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/CastCast}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/rangeRange{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/starttlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/delta*

Tidx0*#
_output_shapes
:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1Castulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range*

DstT0	*

SrcT0*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffListDiffvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2StridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims
ExpandDimslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDenseSparseToDensexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffzlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_values�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:���������
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ReshapeReshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:���������
�
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like	ZerosLikewlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape*
T0	*'
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshapezlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axis*'
_output_shapes
:���������*

Tidx0*
T0	*
N
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ShapeShapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff*
out_type0*
T0	*
_output_shapes
:
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/FillFillulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Shapeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Const*
T0*#
_output_shapes
:���������
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axis*'
_output_shapes
:���������*

Tidx0*
T0	*
N
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axis*#
_output_shapes
:���������*

Tidx0*
T0*
N
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorderSparseReorderxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0*6
_output_shapes$
":���������:���������
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/IdentityIdentitydlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_sliceStridedSlice{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/CastCastlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupGather<linear/text_ids_weighted_by_text_weights/weights/part_0/read}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
validate_indices(*
Tparams0*
Tindices0	*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*'
_output_shapes
:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/RankConst*
dtype0*
value	B :*
_output_shapes
: 
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/subSubvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Rankwlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/y*
T0*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims
ExpandDimsulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/FillFill|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/value*
T0*#
_output_shapes
:���������
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ShapeShapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1*
out_type0*
T0*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Shapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axis*#
_output_shapes
:���������*

Tidx0*
T0*
N
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ReshapeReshapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat*
Tshape0*
T0*'
_output_shapes
:���������
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulMul�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:���������
�
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse
SegmentSumulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:���������
�
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2Reshape{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shape*
Tshape0*
T0
*'
_output_shapes
:���������
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ShapeShapeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
�
mlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_sliceStridedSlice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shapemlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackolinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stackPackalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice*
_output_shapes
:*

axis *
T0*
N
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/TileTileclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:������������������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like	ZerosLikeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
Ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weightsSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tiledlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_likeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/CastCast(read_batch_features/fifo_queue_Dequeue:4*

DstT0*

SrcT0	*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1Slice^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Castglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1ShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/sizeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2Slicealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concatConcatV2alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3ReshapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat*
Tshape0*
T0*'
_output_shapes
:���������
l
linear/linear/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
linear/linear/ReshapeReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3linear/linear/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:���������
�
+linear/bias_weight/part_0/Initializer/ConstConst*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
valueB*    *
_output_shapes
:
�
linear/bias_weight/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
 linear/bias_weight/part_0/AssignAssignlinear/bias_weight/part_0+linear/bias_weight/part_0/Initializer/Const*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
linear/bias_weight/part_0/readIdentitylinear/bias_weight/part_0*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
c
linear/bias_weightIdentitylinear/bias_weight/part_0/read*
T0*
_output_shapes
:
�
linear/linear/BiasAddBiasAddlinear/linear/Reshapelinear/bias_weight*
data_formatNHWC*
T0*'
_output_shapes
:���������
m
predictions/probabilitiesSoftmaxlinear/linear/BiasAdd*
T0*'
_output_shapes
:���������
_
predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
predictions/classesArgMaxlinear/linear/BiasAddpredictions/classes/dimension*

Tidx0*
T0*#
_output_shapes
:���������
�
0training_loss/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*
_output_shapes
:

x
.training_loss/softmax_cross_entropy_loss/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
�
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*$
_output_shapes
:
:

]
training_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
[
train_op/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
]
train_op/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
t
train_op/gradients/FillFilltrain_op/gradients/Shapetrain_op/gradients/Const*
T0*
_output_shapes
: 
}
3train_op/gradients/training_loss_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
-train_op/gradients/training_loss_grad/ReshapeReshapetrain_op/gradients/Fill3train_op/gradients/training_loss_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
~
4train_op/gradients/training_loss_grad/Tile/multiplesConst*
dtype0*
valueB:
*
_output_shapes
:
�
*train_op/gradients/training_loss_grad/TileTile-train_op/gradients/training_loss_grad/Reshape4train_op/gradients/training_loss_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:

u
+train_op/gradients/training_loss_grad/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
p
-train_op/gradients/training_loss_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
u
+train_op/gradients/training_loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
*train_op/gradients/training_loss_grad/ProdProd+train_op/gradients/training_loss_grad/Shape+train_op/gradients/training_loss_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
w
-train_op/gradients/training_loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
,train_op/gradients/training_loss_grad/Prod_1Prod-train_op/gradients/training_loss_grad/Shape_1-train_op/gradients/training_loss_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
q
/train_op/gradients/training_loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
-train_op/gradients/training_loss_grad/MaximumMaximum,train_op/gradients/training_loss_grad/Prod_1/train_op/gradients/training_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
.train_op/gradients/training_loss_grad/floordivFloorDiv*train_op/gradients/training_loss_grad/Prod-train_op/gradients/training_loss_grad/Maximum*
T0*
_output_shapes
: 
�
*train_op/gradients/training_loss_grad/CastCast.train_op/gradients/training_loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
-train_op/gradients/training_loss_grad/truedivRealDiv*train_op/gradients/training_loss_grad/Tile*train_op/gradients/training_loss_grad/Cast*
T0*
_output_shapes
:


train_op/gradients/zeros_like	ZerosLike*training_loss/softmax_cross_entropy_loss:1*
T0*
_output_shapes

:

�
Ptrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/PreventGradientPreventGradient*training_loss/softmax_cross_entropy_loss:1*
T0*
_output_shapes

:

�
Otrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
Ktrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims
ExpandDims-train_op/gradients/training_loss_grad/truedivOtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:

�
Dtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mulMulKtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDimsPtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/PreventGradient*
T0*
_output_shapes

:

�
9train_op/gradients/linear/linear/BiasAdd_grad/BiasAddGradBiasAddGradDtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mul*
data_formatNHWC*
T0*
_output_shapes
:
�
3train_op/gradients/linear/linear/Reshape_grad/ShapeShapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3*
out_type0*
T0*
_output_shapes
:
�
5train_op/gradients/linear/linear/Reshape_grad/ReshapeReshapeDtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mul3train_op/gradients/linear/linear/Reshape_grad/Shape*
Tshape0*
T0*
_output_shapes

:

�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/ShapeShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/ReshapeReshape5train_op/gradients/linear/linear/Reshape_grad/Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Shape*
Tshape0*
T0*
_output_shapes

:

�
|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like	ZerosLikedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like*
T0*'
_output_shapes
:���������
�
xtrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/SelectSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tile�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Reshape|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like*
T0*
_output_shapes

:

�
ztrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/Select_1Select^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tile|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Reshape*
T0*
_output_shapes

:

�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/GatherGatherztrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/Select_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
validate_indices(*
Tparams0*
Tindices0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/ShapeShape�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1Shapeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgsBroadcastGradientArgs�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mulMul�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/Gatherylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/SumSum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/ReshapeReshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul_1Mul�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/Gather*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum_1Sum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Reshape_1Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ShapeConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB"     *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/SizeSize}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
out_type0*
T0	*
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims
ExpandDims�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Size�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_sliceStridedSlice�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Shape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concatConcatV2�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ReshapeReshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat*
Tshape0*
T0*0
_output_shapes
:������������������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape_1Reshape}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims*
Tshape0*
T0	*#
_output_shapes
:���������
�
"train_op/beta1_power/initial_valueConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *fff?*
_output_shapes
: 
�
train_op/beta1_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
train_op/beta1_power/AssignAssigntrain_op/beta1_power"train_op/beta1_power/initial_value*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
train_op/beta1_power/readIdentitytrain_op/beta1_power*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
"train_op/beta2_power/initial_valueConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *w�?*
_output_shapes
: 
�
train_op/beta2_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
train_op/beta2_power/AssignAssigntrain_op/beta2_power"train_op/beta2_power/initial_value*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
train_op/beta2_power/readIdentitytrain_op/beta2_power*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
e
train_op/zerosConst*
dtype0*
valueB	�8*    *
_output_shapes
:	�8
�
<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/AssignAssign<linear/text_ids_weighted_by_text_weights/weights/part_0/Adamtrain_op/zeros*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
Alinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readIdentity<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
g
train_op/zeros_1Const*
dtype0*
valueB	�8*    *
_output_shapes
:	�8
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
Elinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/AssignAssign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1train_op/zeros_1*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readIdentity>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
]
train_op/zeros_2Const*
dtype0*
valueB*    *
_output_shapes
:
�
linear/bias_weight/part_0/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
%linear/bias_weight/part_0/Adam/AssignAssignlinear/bias_weight/part_0/Adamtrain_op/zeros_2*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
#linear/bias_weight/part_0/Adam/readIdentitylinear/bias_weight/part_0/Adam*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
]
train_op/zeros_3Const*
dtype0*
valueB*    *
_output_shapes
:
�
 linear/bias_weight/part_0/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
'linear/bias_weight/part_0/Adam_1/AssignAssign linear/bias_weight/part_0/Adam_1train_op/zeros_3*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
%linear/bias_weight/part_0/Adam_1/readIdentity linear/bias_weight/part_0/Adam_1*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
`
train_op/Adam/learning_rateConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
X
train_op/Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
X
train_op/Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Z
train_op/Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueUnique�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ShapeShapeStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Unique*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
out_type0*
T0	*
_output_shapes
:
�
`train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stackConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB: *
_output_shapes
:
�
btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_1Const*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB:*
_output_shapes
:
�
btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_2Const*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB:*
_output_shapes
:
�
Ztrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_sliceStridedSliceRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Shape`train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stackbtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_1btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSumUnsortedSegmentSum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ReshapeUtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Unique:1Ztrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
Tindices0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/subSubRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub/xtrain_op/beta2_power/read*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Qtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/SqrtSqrtPtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mulMultrain_op/Adam/learning_rateQtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1/xtrain_op/beta1_power/read*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truedivRealDivPtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mulRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2/xtrain_op/Adam/beta1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_1Mul_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSumRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_2MulAlinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readtrain_op/Adam/beta1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign<linear/text_ids_weighted_by_text_weights/weights/part_0/AdamRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_2*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Wtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd
ScatterAddStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
Tindices0	*
use_locking( *
T0*
_output_shapes
:	�8
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_3Mul_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSum_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSum*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3/xtrain_op/Adam/beta2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_4MulRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_3Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_5MulClinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readtrain_op/Adam/beta2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Utrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Assign_1Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_5*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Ytrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1
ScatterAddUtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Assign_1Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_4*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
Tindices0	*
use_locking( *
T0*
_output_shapes
:	�8
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt_1SqrtYtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_6MulTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truedivWtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/addAddStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt_1train_op/Adam/epsilon*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truediv_1RealDivRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_6Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/add*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignSub	AssignSub7linear/text_ids_weighted_by_text_weights/weights/part_0Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truediv_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Wtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_depsNoOpW^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignSubX^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAddZ^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
8train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam	ApplyAdamlinear/bias_weight/part_0linear/bias_weight/part_0/Adam linear/bias_weight/part_0/Adam_1train_op/beta1_power/readtrain_op/beta2_power/readtrain_op/Adam/learning_ratetrain_op/Adam/beta1train_op/Adam/beta2train_op/Adam/epsilon9train_op/gradients/linear/linear/BiasAdd_grad/BiasAddGrad*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking( *
T0*
_output_shapes
:
�
train_op/Adam/mulMultrain_op/beta1_power/readtrain_op/Adam/beta1X^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
train_op/Adam/AssignAssigntrain_op/beta1_powertrain_op/Adam/mul*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
: 
�
train_op/Adam/mul_1Multrain_op/beta2_power/readtrain_op/Adam/beta2X^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
train_op/Adam/Assign_1Assigntrain_op/beta2_powertrain_op/Adam/mul_1*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
: 
�
train_op/Adam/updateNoOpX^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam^train_op/Adam/Assign^train_op/Adam/Assign_1
�
train_op/Adam/valueConst^train_op/Adam/update*
dtype0	*
_class
loc:@global_step*
value	B	 R*
_output_shapes
: 
�
train_op/Adam	AssignAddglobal_steptrain_op/Adam/value*
_class
loc:@global_step*
use_locking( *
T0	*
_output_shapes
: 
�
,metrics/remove_squeezable_dimensions/SqueezeSqueezeExpandDims_1*
squeeze_dims

���������*
T0	*
_output_shapes
:

~
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*
_output_shapes
:

Z
metrics/ToFloatCastmetrics/Equal*

DstT0*

SrcT0
*
_output_shapes
:

[
metrics/accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
validate_shape(*)
_class
loc:@metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
�
metrics/accuracy/total/readIdentitymetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
T0*
_output_shapes
: 
]
metrics/accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
validate_shape(*)
_class
loc:@metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
�
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
T0*
_output_shapes
: 
W
metrics/accuracy/SizeConst*
dtype0*
value	B :
*
_output_shapes
: 
i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
`
metrics/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*)
_class
loc:@metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/accuracy/AssignAdd_1	AssignAddmetrics/accuracy/countmetrics/accuracy/ToFloat_1*)
_class
loc:@metrics/accuracy/count*
use_locking( *
T0*
_output_shapes
: 
_
metrics/accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
}
metrics/accuracy/GreaterGreatermetrics/accuracy/count/readmetrics/accuracy/Greater/y*
T0*
_output_shapes
: 
~
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
T0*
_output_shapes
: 
]
metrics/accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
T0*
_output_shapes
: 
a
metrics/accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
�
metrics/accuracy/truediv_1RealDivmetrics/accuracy/AssignAddmetrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
a
metrics/accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
N
metrics/RankConst*
dtype0*
value	B :*
_output_shapes
: 
U
metrics/LessEqual/yConst*
dtype0*
value	B :*
_output_shapes
: 
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
T0*
_output_shapes
: 
�
metrics/Assert/ConstConst*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
�
metrics/Assert/Assert/data_0Const*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*
	summarize*

T
2
�
metrics/Reshape/shapeConst^metrics/Assert/Assert*
dtype0*
valueB:
���������*
_output_shapes
:
r
metrics/ReshapeReshapeExpandDims_1metrics/Reshape/shape*
Tshape0*
T0	*
_output_shapes
:

]
metrics/one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
^
metrics/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
W
metrics/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
TI0	*
_output_shapes

:
*
T0*
axis���������
]
metrics/CastCastmetrics/one_hot*

DstT0
*

SrcT0*
_output_shapes

:

j
metrics/auc/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:���������
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"   ����*
_output_shapes
:
�
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:	�
d
metrics/auc/ShapeShapemetrics/auc/Reshape*
out_type0*
T0*
_output_shapes
:
i
metrics/auc/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
k
!metrics/auc/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
k
!metrics/auc/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_sliceStridedSlicemetrics/auc/Shapemetrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
�
metrics/auc/ConstConst*
dtype0*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
_output_shapes	
:�
d
metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	�
U
metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
_output_shapes
:*

axis *
T0*
N
�
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:����������
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
T0*
_output_shapes
: 
]
metrics/auc/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
T0*
_output_shapes
:
�
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:���������
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"�      *
_output_shapes
:
�
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
w
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*(
_output_shapes
:����������
c
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*(
_output_shapes
:����������
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"�      *
_output_shapes
:
�
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
* 
_output_shapes
:
��
\
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2* 
_output_shapes
:
��
`
metrics/auc/zerosConst*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/true_positives
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:�
�
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
T0*
_output_shapes	
:�
o
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater* 
_output_shapes
:
��
o
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

DstT0*

SrcT0
* 
_output_shapes
:
��
c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:�
�
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*-
_class#
!loc:@metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_1Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/false_negatives
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
validate_shape(*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:�
�
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
_output_shapes	
:�
t
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot* 
_output_shapes
:
��
q
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:�
�
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_2Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/true_negatives
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:�
�
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
_output_shapes	
:�
z
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot* 
_output_shapes
:
��
q
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:�
�
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_3Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/false_positives
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:�
�
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
T0*
_output_shapes	
:�
w
metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater* 
_output_shapes
:
��
q
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:�
�
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*.
_class$
" loc:@metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:�
V
metrics/auc/add/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:�
�
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_2/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
T0*
_output_shapes	
:�
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
T0*
_output_shapes	
:�
�
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_4/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:�
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:�
Z
metrics/auc/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
T0*
_output_shapes	
:�
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:�
]
metrics/auc/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
|
metrics/auc/valueSummetrics/auc/Mulmetrics/auc/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
X
metrics/auc/add_6/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
T0*
_output_shapes	
:�
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:�
X
metrics/auc/add_8/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:�
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
T0*
_output_shapes	
:�
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:�
Y
metrics/auc/add_10/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:�
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
T0*
_output_shapes	
:�
\
metrics/auc/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
s
metrics/auc/truediv_1RealDivmetrics/auc/add_11metrics/auc/truediv_1/y*
T0*
_output_shapes	
:�
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:�
]
metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
�
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

*metrics/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*
_output_shapes
:

r
(metrics/softmax_cross_entropy_loss/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
�
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*$
_output_shapes
:
:

a
metrics/eval_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: ""�%
cond_context�%�%
�
"read_batch_features/cond/cond_text"read_batch_features/cond/pred_id:0#read_batch_features/cond/switch_t:0 *�
-read_batch_features/cond/control_dependency:0
"read_batch_features/cond/pred_id:0
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch:1
Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1:1
Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2:1
#read_batch_features/cond/switch_t:0
*read_batch_features/random_shuffle_queue:0
+read_batch_features/read/ReaderReadUpToV2:0
+read_batch_features/read/ReaderReadUpToV2:1s
+read_batch_features/read/ReaderReadUpToV2:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2:1s
+read_batch_features/read/ReaderReadUpToV2:0Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1:1p
*read_batch_features/random_shuffle_queue:0Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch:1
�
$read_batch_features/cond/cond_text_1"read_batch_features/cond/pred_id:0#read_batch_features/cond/switch_f:0*z
/read_batch_features/cond/control_dependency_1:0
"read_batch_features/cond/pred_id:0
#read_batch_features/cond/switch_f:0
�
$read_batch_features/cond_1/cond_text$read_batch_features/cond_1/pred_id:0%read_batch_features/cond_1/switch_t:0 *�
/read_batch_features/cond_1/control_dependency:0
$read_batch_features/cond_1/pred_id:0
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch:1
Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1:1
Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2:1
%read_batch_features/cond_1/switch_t:0
*read_batch_features/random_shuffle_queue:0
-read_batch_features/read/ReaderReadUpToV2_1:0
-read_batch_features/read/ReaderReadUpToV2_1:1w
-read_batch_features/read/ReaderReadUpToV2_1:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2:1w
-read_batch_features/read/ReaderReadUpToV2_1:0Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1:1r
*read_batch_features/random_shuffle_queue:0Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch:1
�
&read_batch_features/cond_1/cond_text_1$read_batch_features/cond_1/pred_id:0%read_batch_features/cond_1/switch_f:0*�
1read_batch_features/cond_1/control_dependency_1:0
$read_batch_features/cond_1/pred_id:0
%read_batch_features/cond_1/switch_f:0
�
$read_batch_features/cond_2/cond_text$read_batch_features/cond_2/pred_id:0%read_batch_features/cond_2/switch_t:0 *�
/read_batch_features/cond_2/control_dependency:0
$read_batch_features/cond_2/pred_id:0
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch:1
Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1:1
Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2:1
%read_batch_features/cond_2/switch_t:0
*read_batch_features/random_shuffle_queue:0
-read_batch_features/read/ReaderReadUpToV2_2:0
-read_batch_features/read/ReaderReadUpToV2_2:1w
-read_batch_features/read/ReaderReadUpToV2_2:0Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1:1w
-read_batch_features/read/ReaderReadUpToV2_2:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2:1r
*read_batch_features/random_shuffle_queue:0Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch:1
�
&read_batch_features/cond_2/cond_text_1$read_batch_features/cond_2/pred_id:0%read_batch_features/cond_2/switch_f:0*�
1read_batch_features/cond_2/control_dependency_1:0
$read_batch_features/cond_2/pred_id:0
%read_batch_features/cond_2/switch_f:0
�
$read_batch_features/cond_3/cond_text$read_batch_features/cond_3/pred_id:0%read_batch_features/cond_3/switch_t:0 *�
/read_batch_features/cond_3/control_dependency:0
$read_batch_features/cond_3/pred_id:0
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch:1
Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1:1
Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2:1
%read_batch_features/cond_3/switch_t:0
*read_batch_features/random_shuffle_queue:0
-read_batch_features/read/ReaderReadUpToV2_3:0
-read_batch_features/read/ReaderReadUpToV2_3:1w
-read_batch_features/read/ReaderReadUpToV2_3:0Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1:1w
-read_batch_features/read/ReaderReadUpToV2_3:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2:1r
*read_batch_features/random_shuffle_queue:0Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch:1
�
&read_batch_features/cond_3/cond_text_1$read_batch_features/cond_3/pred_id:0%read_batch_features/cond_3/switch_f:0*�
1read_batch_features/cond_3/control_dependency_1:0
$read_batch_features/cond_3/pred_id:0
%read_batch_features/cond_3/switch_f:0" 
global_step

global_step:0"�
trainable_variables��
�
9linear/text_ids_weighted_by_text_weights/weights/part_0:0>linear/text_ids_weighted_by_text_weights/weights/part_0/Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/read:0"@
0linear/text_ids_weighted_by_text_weights/weights�8  "�8
�
linear/bias_weight/part_0:0 linear/bias_weight/part_0/Assign linear/bias_weight/part_0/read:0"
linear/bias_weight ""�
	variables��
7
global_step:0global_step/Assignglobal_step/read:0
�
9linear/text_ids_weighted_by_text_weights/weights/part_0:0>linear/text_ids_weighted_by_text_weights/weights/part_0/Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/read:0"@
0linear/text_ids_weighted_by_text_weights/weights�8  "�8
�
linear/bias_weight/part_0:0 linear/bias_weight/part_0/Assign linear/bias_weight/part_0/read:0"
linear/bias_weight "
R
train_op/beta1_power:0train_op/beta1_power/Assigntrain_op/beta1_power/read:0
R
train_op/beta2_power:0train_op/beta2_power/Assigntrain_op/beta2_power/read:0
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam:0Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/AssignClinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/read:0"E
5linear/text_ids_weighted_by_text_weights/weights/Adam�8  "�8
�
@linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1:0Elinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/AssignElinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/read:0"G
7linear/text_ids_weighted_by_text_weights/weights/Adam_1�8  "�8
�
 linear/bias_weight/part_0/Adam:0%linear/bias_weight/part_0/Adam/Assign%linear/bias_weight/part_0/Adam/read:0""
linear/bias_weight/Adam "
�
"linear/bias_weight/part_0/Adam_1:0'linear/bias_weight/part_0/Adam_1/Assign'linear/bias_weight/part_0/Adam_1/read:0"$
linear/bias_weight/Adam_1 ""
losses

training_loss:0"
train_op

train_op/Adam"�
local_variables�
�
metrics/accuracy/total:0
metrics/accuracy/count:0
metrics/auc/true_positives:0
metrics/auc/false_negatives:0
metrics/auc/true_negatives:0
metrics/auc/false_positives:0
metrics/mean/total:0
metrics/mean/count:0"�
queue_runners��
�
#read_batch_features/file_name_queue?read_batch_features/file_name_queue/file_name_queue_EnqueueMany9read_batch_features/file_name_queue/file_name_queue_Close";read_batch_features/file_name_queue/file_name_queue_Close_1*
�
(read_batch_features/random_shuffle_queue read_batch_features/cond/Merge:0"read_batch_features/cond_1/Merge:0"read_batch_features/cond_2/Merge:0"read_batch_features/cond_3/Merge:0.read_batch_features/random_shuffle_queue_Close"0read_batch_features/random_shuffle_queue_Close_1*
�
read_batch_features/fifo_queue&read_batch_features/fifo_queue_enqueue(read_batch_features/fifo_queue_enqueue_1$read_batch_features/fifo_queue_Close"&read_batch_features/fifo_queue_Close_1*"

savers "�
	summaries�
�
9read_batch_features/file_name_queue/fraction_of_32_full:0
1read_batch_features/fraction_over_10_of_10_full:0
_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full:0
training_loss/ScalarSummary:0"m
model_variablesZ
X
9linear/text_ids_weighted_by_text_weights/weights/part_0:0
linear/bias_weight/part_0:0"d
linearZ
X
9linear/text_ids_weighted_by_text_weights/weights/part_0:0
linear/bias_weight/part_0:0n���M�     �p��	2����@�A"��


global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
�
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
�
)read_batch_features/file_name_queue/inputConst*
dtype0*�
value�B�B/exout/features_train-00011-of-00014.tfrecord.gzB/exout/features_train-00013-of-00014.tfrecord.gzB/exout/features_train-00001-of-00014.tfrecord.gzB/exout/features_train-00006-of-00014.tfrecord.gzB/exout/features_train-00010-of-00014.tfrecord.gzB/exout/features_train-00008-of-00014.tfrecord.gzB/exout/features_train-00004-of-00014.tfrecord.gzB/exout/features_train-00005-of-00014.tfrecord.gzB/exout/features_train-00009-of-00014.tfrecord.gzB/exout/features_train-00012-of-00014.tfrecord.gzB/exout/features_train-00000-of-00014.tfrecord.gzB/exout/features_train-00003-of-00014.tfrecord.gzB/exout/features_train-00002-of-00014.tfrecord.gzB/exout/features_train-00007-of-00014.tfrecord.gz*
_output_shapes
:
j
(read_batch_features/file_name_queue/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
o
-read_batch_features/file_name_queue/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
�
+read_batch_features/file_name_queue/GreaterGreater(read_batch_features/file_name_queue/Size-read_batch_features/file_name_queue/Greater/y*
T0*
_output_shapes
: 
�
0read_batch_features/file_name_queue/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
�
8read_batch_features/file_name_queue/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
�
1read_batch_features/file_name_queue/Assert/AssertAssert+read_batch_features/file_name_queue/Greater8read_batch_features/file_name_queue/Assert/Assert/data_0*
	summarize*

T
2
�
,read_batch_features/file_name_queue/IdentityIdentity)read_batch_features/file_name_queue/input2^read_batch_features/file_name_queue/Assert/Assert*
T0*
_output_shapes
:
�
1read_batch_features/file_name_queue/RandomShuffleRandomShuffle,read_batch_features/file_name_queue/Identity*
seed2 *

seed *
T0*
_output_shapes
:
�
#read_batch_features/file_name_queueFIFOQueueV2*
capacity *
_output_shapes
: *
shapes
: *
component_types
2*
	container *
shared_name 
�
?read_batch_features/file_name_queue/file_name_queue_EnqueueManyQueueEnqueueManyV2#read_batch_features/file_name_queue1read_batch_features/file_name_queue/RandomShuffle*

timeout_ms���������*
Tcomponents
2
�
9read_batch_features/file_name_queue/file_name_queue_CloseQueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues( 
�
;read_batch_features/file_name_queue/file_name_queue_Close_1QueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues(
�
8read_batch_features/file_name_queue/file_name_queue_SizeQueueSizeV2#read_batch_features/file_name_queue*
_output_shapes
: 
�
(read_batch_features/file_name_queue/CastCast8read_batch_features/file_name_queue/file_name_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
n
)read_batch_features/file_name_queue/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
�
'read_batch_features/file_name_queue/mulMul(read_batch_features/file_name_queue/Cast)read_batch_features/file_name_queue/mul/y*
T0*
_output_shapes
: 
�
<read_batch_features/file_name_queue/fraction_of_32_full/tagsConst*
dtype0*H
value?B= B7read_batch_features/file_name_queue/fraction_of_32_full*
_output_shapes
: 
�
7read_batch_features/file_name_queue/fraction_of_32_fullScalarSummary<read_batch_features/file_name_queue/fraction_of_32_full/tags'read_batch_features/file_name_queue/mul*
T0*
_output_shapes
: 
�
)read_batch_features/read/TFRecordReaderV2TFRecordReaderV2*
	container *
shared_name *
compression_typeGZIP*
_output_shapes
: 
w
5read_batch_features/read/ReaderReadUpToV2/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
)read_batch_features/read/ReaderReadUpToV2ReaderReadUpToV2)read_batch_features/read/TFRecordReaderV2#read_batch_features/file_name_queue5read_batch_features/read/ReaderReadUpToV2/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_1TFRecordReaderV2*
	container *
shared_name *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_1/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_1ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_1#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_1/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_2TFRecordReaderV2*
	container *
shared_name *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_2/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_2ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_2#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_2/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_3TFRecordReaderV2*
	container *
shared_name *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_3/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_3ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_3#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_3/num_records*2
_output_shapes 
:���������:���������
[
read_batch_features/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
(read_batch_features/random_shuffle_queueRandomShuffleQueueV2*
capacity*
component_types
2*
_output_shapes
: *
min_after_dequeue
*
shapes
: : *
seed2 *

seed *
	container *
shared_name 
�
read_batch_features/cond/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
q
!read_batch_features/cond/switch_tIdentity!read_batch_features/cond/Switch:1*
T0
*
_output_shapes
: 
o
!read_batch_features/cond/switch_fIdentityread_batch_features/cond/Switch*
T0
*
_output_shapes
: 
h
 read_batch_features/cond/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
@read_batch_features/cond/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue read_batch_features/cond/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1Switch)read_batch_features/read/ReaderReadUpToV2 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:���������:���������
�
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2Switch+read_batch_features/read/ReaderReadUpToV2:1 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:���������:���������
�
9read_batch_features/cond/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
+read_batch_features/cond/control_dependencyIdentity!read_batch_features/cond/switch_t:^read_batch_features/cond/random_shuffle_queue_EnqueueMany*4
_class*
(&loc:@read_batch_features/cond/switch_t*
T0
*
_output_shapes
: 
I
read_batch_features/cond/NoOpNoOp"^read_batch_features/cond/switch_f
�
-read_batch_features/cond/control_dependency_1Identity!read_batch_features/cond/switch_f^read_batch_features/cond/NoOp*4
_class*
(&loc:@read_batch_features/cond/switch_f*
T0
*
_output_shapes
: 
�
read_batch_features/cond/MergeMerge-read_batch_features/cond/control_dependency_1+read_batch_features/cond/control_dependency*
_output_shapes
: : *
T0
*
N
�
!read_batch_features/cond_1/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_1/switch_tIdentity#read_batch_features/cond_1/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_1/switch_fIdentity!read_batch_features/cond_1/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_1/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_1/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_1:1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_1/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_1/control_dependencyIdentity#read_batch_features/cond_1/switch_t<^read_batch_features/cond_1/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_1/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_1/NoOpNoOp$^read_batch_features/cond_1/switch_f
�
/read_batch_features/cond_1/control_dependency_1Identity#read_batch_features/cond_1/switch_f ^read_batch_features/cond_1/NoOp*6
_class,
*(loc:@read_batch_features/cond_1/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_1/MergeMerge/read_batch_features/cond_1/control_dependency_1-read_batch_features/cond_1/control_dependency*
_output_shapes
: : *
T0
*
N
�
!read_batch_features/cond_2/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_2/switch_tIdentity#read_batch_features/cond_2/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_2/switch_fIdentity!read_batch_features/cond_2/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_2/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_2/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_2"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_2:1"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_2/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_2/control_dependencyIdentity#read_batch_features/cond_2/switch_t<^read_batch_features/cond_2/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_2/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_2/NoOpNoOp$^read_batch_features/cond_2/switch_f
�
/read_batch_features/cond_2/control_dependency_1Identity#read_batch_features/cond_2/switch_f ^read_batch_features/cond_2/NoOp*6
_class,
*(loc:@read_batch_features/cond_2/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_2/MergeMerge/read_batch_features/cond_2/control_dependency_1-read_batch_features/cond_2/control_dependency*
_output_shapes
: : *
T0
*
N
�
!read_batch_features/cond_3/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_3/switch_tIdentity#read_batch_features/cond_3/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_3/switch_fIdentity!read_batch_features/cond_3/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_3/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_3/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_3"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_3:1"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_3/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_3/control_dependencyIdentity#read_batch_features/cond_3/switch_t<^read_batch_features/cond_3/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_3/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_3/NoOpNoOp$^read_batch_features/cond_3/switch_f
�
/read_batch_features/cond_3/control_dependency_1Identity#read_batch_features/cond_3/switch_f ^read_batch_features/cond_3/NoOp*6
_class,
*(loc:@read_batch_features/cond_3/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_3/MergeMerge/read_batch_features/cond_3/control_dependency_1-read_batch_features/cond_3/control_dependency*
_output_shapes
: : *
T0
*
N
�
.read_batch_features/random_shuffle_queue_CloseQueueCloseV2(read_batch_features/random_shuffle_queue*
cancel_pending_enqueues( 
�
0read_batch_features/random_shuffle_queue_Close_1QueueCloseV2(read_batch_features/random_shuffle_queue*
cancel_pending_enqueues(
~
-read_batch_features/random_shuffle_queue_SizeQueueSizeV2(read_batch_features/random_shuffle_queue*
_output_shapes
: 
[
read_batch_features/sub/yConst*
dtype0*
value	B :
*
_output_shapes
: 
�
read_batch_features/subSub-read_batch_features/random_shuffle_queue_Sizeread_batch_features/sub/y*
T0*
_output_shapes
: 
_
read_batch_features/Maximum/xConst*
dtype0*
value	B : *
_output_shapes
: 

read_batch_features/MaximumMaximumread_batch_features/Maximum/xread_batch_features/sub*
T0*
_output_shapes
: 
m
read_batch_features/CastCastread_batch_features/Maximum*

DstT0*

SrcT0*
_output_shapes
: 
^
read_batch_features/mul/yConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
t
read_batch_features/mulMulread_batch_features/Castread_batch_features/mul/y*
T0*
_output_shapes
: 
�
4read_batch_features/fraction_over_10_of_10_full/tagsConst*
dtype0*@
value7B5 B/read_batch_features/fraction_over_10_of_10_full*
_output_shapes
: 
�
/read_batch_features/fraction_over_10_of_10_fullScalarSummary4read_batch_features/fraction_over_10_of_10_full/tagsread_batch_features/mul*
T0*
_output_shapes
: 
W
read_batch_features/nConst*
dtype0*
value	B :
*
_output_shapes
: 
�
read_batch_featuresQueueDequeueManyV2(read_batch_features/random_shuffle_queueread_batch_features/n*

timeout_ms���������*
component_types
2* 
_output_shapes
:
:

j
(read_batch_features/ParseExample/key_keyConst*
dtype0	*
value	B	 R *
_output_shapes
: 
q
.read_batch_features/ParseExample/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
(read_batch_features/ParseExample/ReshapeReshape(read_batch_features/ParseExample/key_key.read_batch_features/ParseExample/Reshape/shape*
_output_shapes
: *
T0	*
Tshape0
i
&read_batch_features/ParseExample/ConstConst*
dtype0	*
valueB	 *
_output_shapes
: 
v
3read_batch_features/ParseExample/ParseExample/namesConst*
dtype0*
valueB *
_output_shapes
: 
�
;read_batch_features/ParseExample/ParseExample/sparse_keys_0Const*
dtype0*
valueB Btext_ids*
_output_shapes
: 
�
;read_batch_features/ParseExample/ParseExample/sparse_keys_1Const*
dtype0*
valueB Btext_weights*
_output_shapes
: 
~
:read_batch_features/ParseExample/ParseExample/dense_keys_0Const*
dtype0*
valueB	 Bkey*
_output_shapes
: 
�
:read_batch_features/ParseExample/ParseExample/dense_keys_1Const*
dtype0*
valueB Btarget*
_output_shapes
: 
�
-read_batch_features/ParseExample/ParseExampleParseExampleread_batch_features:13read_batch_features/ParseExample/ParseExample/names;read_batch_features/ParseExample/ParseExample/sparse_keys_0;read_batch_features/ParseExample/ParseExample/sparse_keys_1:read_batch_features/ParseExample/ParseExample/dense_keys_0:read_batch_features/ParseExample/ParseExample/dense_keys_1(read_batch_features/ParseExample/Reshape&read_batch_features/ParseExample/Const*
dense_shapes
: : *p
_output_shapes^
\:���������:���������:���������:���������:::
:
*
Ndense*
sparse_types
2	*
Tdense
2		*
Nsparse
�
read_batch_features/fifo_queueFIFOQueueV2*
capacityd*
_output_shapes
: *
shapes
 * 
component_types
2								*
	container *
shared_name 
j
#read_batch_features/fifo_queue_SizeQueueSizeV2read_batch_features/fifo_queue*
_output_shapes
: 
w
read_batch_features/Cast_1Cast#read_batch_features/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
`
read_batch_features/mul_1/yConst*
dtype0*
valueB
 *
�#<*
_output_shapes
: 
z
read_batch_features/mul_1Mulread_batch_features/Cast_1read_batch_features/mul_1/y*
T0*
_output_shapes
: 
�
bread_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full/tagsConst*
dtype0*n
valueeBc B]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full*
_output_shapes
: 
�
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_fullScalarSummarybread_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full/tagsread_batch_features/mul_1*
T0*
_output_shapes
: 
�
&read_batch_features/fifo_queue_enqueueQueueEnqueueV2read_batch_features/fifo_queue/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms���������*
Tcomponents
2								
�
(read_batch_features/fifo_queue_enqueue_1QueueEnqueueV2read_batch_features/fifo_queue/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms���������*
Tcomponents
2								
s
$read_batch_features/fifo_queue_CloseQueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues( 
u
&read_batch_features/fifo_queue_Close_1QueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues(
�
&read_batch_features/fifo_queue_DequeueQueueDequeueV2read_batch_features/fifo_queue*

timeout_ms���������* 
component_types
2								*v
_output_shapesd
b:
:
:���������:���������::���������:���������::

Y
ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�

ExpandDims
ExpandDims&read_batch_features/fifo_queue_DequeueExpandDims/dim*

Tdim0*
T0	*
_output_shapes

:

[
ExpandDims_1/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
ExpandDims_1
ExpandDims(read_batch_features/fifo_queue_Dequeue:1ExpandDims_1/dim*

Tdim0*
T0	*
_output_shapes

:

V
linear/linear/mod/yConst*
dtype0	*
value
B	 R�8*
_output_shapes
: 
�
linear/linear/modFloorMod(read_batch_features/fifo_queue_Dequeue:3linear/linear/mod/y*
T0	*#
_output_shapes
:���������
�
Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/ConstConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB	�8*    *
_output_shapes
:	�8
�
7linear/text_ids_weighted_by_text_weights/weights/part_0
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign7linear/text_ids_weighted_by_text_weights/weights/part_0Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/Const*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
<linear/text_ids_weighted_by_text_weights/weights/part_0/readIdentity7linear/text_ids_weighted_by_text_weights/weights/part_0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SliceSlice(read_batch_features/fifo_queue_Dequeue:4elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/begindlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ProdProd_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
�
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GatherGather(read_batch_features/fifo_queue_Dequeue:4hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
�
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shapePack^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Prod`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather*
N*
T0	*
_output_shapes
:*

axis 
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeSparseReshape(read_batch_features/fifo_queue_Dequeue:2(read_batch_features/fifo_queue_Dequeue:4qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shape*-
_output_shapes
:���������:
�
plinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/IdentityIdentitylinear/linear/mod*
T0	*#
_output_shapes
:���������
�
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualGreaterEqualplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityhlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/y*
T0	*#
_output_shapes
:���������
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterGreater(read_batch_features/fifo_queue_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/y*
T0*#
_output_shapes
:���������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd
LogicalAndflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater*#
_output_shapes
:���������
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/WhereWheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:���������
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ReshapeReshape_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Whereglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shape*#
_output_shapes
:���������*
T0	*
Tshape0
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2Gatherplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityIdentityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1Wheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:���������
�
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1Reshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shape*#
_output_shapes
:���������*
T0	*
Tshape0
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4Gather(read_batch_features/fifo_queue_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Identityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_sliceStridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/CastCast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/rangeRangeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startrlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Castylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:���������
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1Castslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffListDifftlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims
ExpandDims}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseSparseToDensevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_values�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ReshapeReshapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shape*'
_output_shapes
:���������*
T0	*
Tshape0
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_like	ZerosLikeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:���������
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concatConcatV2ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_likeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0	
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ShapeShapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
�
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/FillFillslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Shapeslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0	
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Fill{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axis*
N*

Tidx0*#
_output_shapes
:���������*
T0	
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorderSparseReordervlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*6
_output_shapes$
":���������:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/IdentityIdentityblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*
_output_shapes
:
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_sliceStridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/CastCast}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/rangeRange{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/starttlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/delta*

Tidx0*#
_output_shapes
:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1Castulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range*

DstT0	*

SrcT0*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffListDiffvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2StridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims
ExpandDimslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDenseSparseToDensexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffzlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_values�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:���������
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ReshapeReshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shape*'
_output_shapes
:���������*
T0	*
Tshape0
�
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like	ZerosLikewlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape*
T0	*'
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshapezlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0	
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ShapeShapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff*
out_type0*
T0	*
_output_shapes
:
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/FillFillulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Shapeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Const*
T0*#
_output_shapes
:���������
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axis*
N*

Tidx0*'
_output_shapes
:���������*
T0	
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axis*
N*

Tidx0*#
_output_shapes
:���������*
T0
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorderSparseReorderxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0*6
_output_shapes$
":���������:���������
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/IdentityIdentitydlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_sliceStridedSlice{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/CastCastlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupGather<linear/text_ids_weighted_by_text_weights/weights/part_0/read}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
validate_indices(*
Tparams0*
Tindices0	*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*'
_output_shapes
:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/RankConst*
dtype0*
value	B :*
_output_shapes
: 
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/subSubvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Rankwlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/y*
T0*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims
ExpandDimsulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/FillFill|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/value*
T0*#
_output_shapes
:���������
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ShapeShapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1*
out_type0*
T0*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Shapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axis*
N*

Tidx0*#
_output_shapes
:���������*
T0
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ReshapeReshapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat*'
_output_shapes
:���������*
T0*
Tshape0
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulMul�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:���������
�
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse
SegmentSumulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:���������
�
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2Reshape{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shape*'
_output_shapes
:���������*
T0
*
Tshape0
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ShapeShapeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
�
mlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_sliceStridedSlice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shapemlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackolinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stackPackalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/TileTileclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:������������������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like	ZerosLikeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
Ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weightsSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tiledlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_likeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/CastCast(read_batch_features/fifo_queue_Dequeue:4*

DstT0*

SrcT0	*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1Slice^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Castglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1ShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/sizeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2Slicealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concatConcatV2alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3ReshapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat*'
_output_shapes
:���������*
T0*
Tshape0
l
linear/linear/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
linear/linear/ReshapeReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3linear/linear/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
+linear/bias_weight/part_0/Initializer/ConstConst*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
valueB*    *
_output_shapes
:
�
linear/bias_weight/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
 linear/bias_weight/part_0/AssignAssignlinear/bias_weight/part_0+linear/bias_weight/part_0/Initializer/Const*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
linear/bias_weight/part_0/readIdentitylinear/bias_weight/part_0*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
c
linear/bias_weightIdentitylinear/bias_weight/part_0/read*
T0*
_output_shapes
:
�
linear/linear/BiasAddBiasAddlinear/linear/Reshapelinear/bias_weight*'
_output_shapes
:���������*
T0*
data_formatNHWC
m
predictions/probabilitiesSoftmaxlinear/linear/BiasAdd*
T0*'
_output_shapes
:���������
_
predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
predictions/classesArgMaxlinear/linear/BiasAddpredictions/classes/dimension*#
_output_shapes
:���������*
T0*

Tidx0
�
0training_loss/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*
_output_shapes
:

x
.training_loss/softmax_cross_entropy_loss/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
�
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*$
_output_shapes
:
:

]
training_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
[
train_op/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
]
train_op/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
t
train_op/gradients/FillFilltrain_op/gradients/Shapetrain_op/gradients/Const*
T0*
_output_shapes
: 
}
3train_op/gradients/training_loss_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
-train_op/gradients/training_loss_grad/ReshapeReshapetrain_op/gradients/Fill3train_op/gradients/training_loss_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
~
4train_op/gradients/training_loss_grad/Tile/multiplesConst*
dtype0*
valueB:
*
_output_shapes
:
�
*train_op/gradients/training_loss_grad/TileTile-train_op/gradients/training_loss_grad/Reshape4train_op/gradients/training_loss_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:

u
+train_op/gradients/training_loss_grad/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
p
-train_op/gradients/training_loss_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
u
+train_op/gradients/training_loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
*train_op/gradients/training_loss_grad/ProdProd+train_op/gradients/training_loss_grad/Shape+train_op/gradients/training_loss_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
w
-train_op/gradients/training_loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
,train_op/gradients/training_loss_grad/Prod_1Prod-train_op/gradients/training_loss_grad/Shape_1-train_op/gradients/training_loss_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
q
/train_op/gradients/training_loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
-train_op/gradients/training_loss_grad/MaximumMaximum,train_op/gradients/training_loss_grad/Prod_1/train_op/gradients/training_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
.train_op/gradients/training_loss_grad/floordivFloorDiv*train_op/gradients/training_loss_grad/Prod-train_op/gradients/training_loss_grad/Maximum*
T0*
_output_shapes
: 
�
*train_op/gradients/training_loss_grad/CastCast.train_op/gradients/training_loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
-train_op/gradients/training_loss_grad/truedivRealDiv*train_op/gradients/training_loss_grad/Tile*train_op/gradients/training_loss_grad/Cast*
T0*
_output_shapes
:


train_op/gradients/zeros_like	ZerosLike*training_loss/softmax_cross_entropy_loss:1*
T0*
_output_shapes

:

�
Ptrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/PreventGradientPreventGradient*training_loss/softmax_cross_entropy_loss:1*
T0*
_output_shapes

:

�
Otrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
Ktrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims
ExpandDims-train_op/gradients/training_loss_grad/truedivOtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:

�
Dtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mulMulKtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDimsPtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/PreventGradient*
T0*
_output_shapes

:

�
9train_op/gradients/linear/linear/BiasAdd_grad/BiasAddGradBiasAddGradDtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mul*
_output_shapes
:*
T0*
data_formatNHWC
�
3train_op/gradients/linear/linear/Reshape_grad/ShapeShapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3*
out_type0*
T0*
_output_shapes
:
�
5train_op/gradients/linear/linear/Reshape_grad/ReshapeReshapeDtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mul3train_op/gradients/linear/linear/Reshape_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/ShapeShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/ReshapeReshape5train_op/gradients/linear/linear/Reshape_grad/Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like	ZerosLikedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like*
T0*'
_output_shapes
:���������
�
xtrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/SelectSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tile�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Reshape|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like*
T0*
_output_shapes

:

�
ztrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/Select_1Select^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tile|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Reshape*
T0*
_output_shapes

:

�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/GatherGatherztrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/Select_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
validate_indices(*
Tparams0*
Tindices0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/ShapeShape�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1Shapeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgsBroadcastGradientArgs�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mulMul�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/Gatherylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/SumSum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/ReshapeReshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul_1Mul�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/Gather*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum_1Sum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Reshape_1Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ShapeConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB"     *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/SizeSize}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
out_type0*
T0	*
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims
ExpandDims�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Size�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_sliceStridedSlice�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Shape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concatConcatV2�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ReshapeReshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat*0
_output_shapes
:������������������*
T0*
Tshape0
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape_1Reshape}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims*#
_output_shapes
:���������*
T0	*
Tshape0
�
"train_op/beta1_power/initial_valueConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *fff?*
_output_shapes
: 
�
train_op/beta1_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
train_op/beta1_power/AssignAssigntrain_op/beta1_power"train_op/beta1_power/initial_value*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
train_op/beta1_power/readIdentitytrain_op/beta1_power*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
"train_op/beta2_power/initial_valueConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *w�?*
_output_shapes
: 
�
train_op/beta2_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
train_op/beta2_power/AssignAssigntrain_op/beta2_power"train_op/beta2_power/initial_value*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
train_op/beta2_power/readIdentitytrain_op/beta2_power*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
e
train_op/zerosConst*
dtype0*
valueB	�8*    *
_output_shapes
:	�8
�
<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/AssignAssign<linear/text_ids_weighted_by_text_weights/weights/part_0/Adamtrain_op/zeros*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
Alinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readIdentity<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
g
train_op/zeros_1Const*
dtype0*
valueB	�8*    *
_output_shapes
:	�8
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
Elinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/AssignAssign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1train_op/zeros_1*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readIdentity>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
]
train_op/zeros_2Const*
dtype0*
valueB*    *
_output_shapes
:
�
linear/bias_weight/part_0/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
%linear/bias_weight/part_0/Adam/AssignAssignlinear/bias_weight/part_0/Adamtrain_op/zeros_2*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
#linear/bias_weight/part_0/Adam/readIdentitylinear/bias_weight/part_0/Adam*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
]
train_op/zeros_3Const*
dtype0*
valueB*    *
_output_shapes
:
�
 linear/bias_weight/part_0/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
'linear/bias_weight/part_0/Adam_1/AssignAssign linear/bias_weight/part_0/Adam_1train_op/zeros_3*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
%linear/bias_weight/part_0/Adam_1/readIdentity linear/bias_weight/part_0/Adam_1*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
`
train_op/Adam/learning_rateConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
X
train_op/Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
X
train_op/Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Z
train_op/Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueUnique�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ShapeShapeStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Unique*
out_type0*
T0	*
_output_shapes
:*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
`train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stackConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB: *
_output_shapes
:
�
btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_1Const*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB:*
_output_shapes
:
�
btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_2Const*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB:*
_output_shapes
:
�
Ztrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_sliceStridedSliceRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Shape`train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stackbtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_1btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSumUnsortedSegmentSum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ReshapeUtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Unique:1Ztrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice*
Tindices0*
T0*0
_output_shapes
:������������������*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/subSubRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub/xtrain_op/beta2_power/read*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Qtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/SqrtSqrtPtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mulMultrain_op/Adam/learning_rateQtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1/xtrain_op/beta1_power/read*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truedivRealDivPtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mulRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2/xtrain_op/Adam/beta1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_1Mul_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSumRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_2MulAlinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readtrain_op/Adam/beta1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign<linear/text_ids_weighted_by_text_weights/weights/part_0/AdamRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_2*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Wtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd
ScatterAddStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_1*
Tindices0	*
use_locking( *
T0*
_output_shapes
:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_3Mul_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSum_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSum*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3/xtrain_op/Adam/beta2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_4MulRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_3Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_5MulClinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readtrain_op/Adam/beta2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Utrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Assign_1Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_5*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Ytrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1
ScatterAddUtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Assign_1Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_4*
Tindices0	*
use_locking( *
T0*
_output_shapes
:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt_1SqrtYtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_6MulTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truedivWtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/addAddStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt_1train_op/Adam/epsilon*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truediv_1RealDivRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_6Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/add*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignSub	AssignSub7linear/text_ids_weighted_by_text_weights/weights/part_0Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truediv_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Wtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_depsNoOpW^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignSubX^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAddZ^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
8train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam	ApplyAdamlinear/bias_weight/part_0linear/bias_weight/part_0/Adam linear/bias_weight/part_0/Adam_1train_op/beta1_power/readtrain_op/beta2_power/readtrain_op/Adam/learning_ratetrain_op/Adam/beta1train_op/Adam/beta2train_op/Adam/epsilon9train_op/gradients/linear/linear/BiasAdd_grad/BiasAddGrad*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking( *
T0*
_output_shapes
:
�
train_op/Adam/mulMultrain_op/beta1_power/readtrain_op/Adam/beta1X^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
train_op/Adam/AssignAssigntrain_op/beta1_powertrain_op/Adam/mul*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
: 
�
train_op/Adam/mul_1Multrain_op/beta2_power/readtrain_op/Adam/beta2X^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
train_op/Adam/Assign_1Assigntrain_op/beta2_powertrain_op/Adam/mul_1*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
: 
�
train_op/Adam/updateNoOpX^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam^train_op/Adam/Assign^train_op/Adam/Assign_1
�
train_op/Adam/valueConst^train_op/Adam/update*
dtype0	*
_class
loc:@global_step*
value	B	 R*
_output_shapes
: 
�
train_op/Adam	AssignAddglobal_steptrain_op/Adam/value*
_class
loc:@global_step*
use_locking( *
T0	*
_output_shapes
: 
�
,metrics/remove_squeezable_dimensions/SqueezeSqueezeExpandDims_1*
squeeze_dims

���������*
T0	*
_output_shapes
:

~
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*
_output_shapes
:

Z
metrics/ToFloatCastmetrics/Equal*

DstT0*

SrcT0
*
_output_shapes
:

[
metrics/accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
validate_shape(*)
_class
loc:@metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
�
metrics/accuracy/total/readIdentitymetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
T0*
_output_shapes
: 
]
metrics/accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
validate_shape(*)
_class
loc:@metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
�
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
T0*
_output_shapes
: 
W
metrics/accuracy/SizeConst*
dtype0*
value	B :
*
_output_shapes
: 
i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
`
metrics/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*)
_class
loc:@metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/accuracy/AssignAdd_1	AssignAddmetrics/accuracy/countmetrics/accuracy/ToFloat_1*)
_class
loc:@metrics/accuracy/count*
use_locking( *
T0*
_output_shapes
: 
_
metrics/accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
}
metrics/accuracy/GreaterGreatermetrics/accuracy/count/readmetrics/accuracy/Greater/y*
T0*
_output_shapes
: 
~
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
T0*
_output_shapes
: 
]
metrics/accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
T0*
_output_shapes
: 
a
metrics/accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
�
metrics/accuracy/truediv_1RealDivmetrics/accuracy/AssignAddmetrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
a
metrics/accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
N
metrics/RankConst*
dtype0*
value	B :*
_output_shapes
: 
U
metrics/LessEqual/yConst*
dtype0*
value	B :*
_output_shapes
: 
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
T0*
_output_shapes
: 
�
metrics/Assert/ConstConst*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
�
metrics/Assert/Assert/data_0Const*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*
	summarize*

T
2
�
metrics/Reshape/shapeConst^metrics/Assert/Assert*
dtype0*
valueB:
���������*
_output_shapes
:
r
metrics/ReshapeReshapeExpandDims_1metrics/Reshape/shape*
_output_shapes
:
*
T0	*
Tshape0
]
metrics/one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
^
metrics/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
W
metrics/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
axis���������*
T0*
_output_shapes

:
*
TI0	
]
metrics/CastCastmetrics/one_hot*

DstT0
*

SrcT0*
_output_shapes

:

j
metrics/auc/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"   ����*
_output_shapes
:
�
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
_output_shapes
:	�*
T0
*
Tshape0
d
metrics/auc/ShapeShapemetrics/auc/Reshape*
out_type0*
T0*
_output_shapes
:
i
metrics/auc/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
k
!metrics/auc/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
k
!metrics/auc/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_sliceStridedSlicemetrics/auc/Shapemetrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
�
metrics/auc/ConstConst*
dtype0*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
_output_shapes	
:�
d
metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	�
U
metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
N*
T0*
_output_shapes
:*

axis 
�
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:����������
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
T0*
_output_shapes
: 
]
metrics/auc/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
T0*
_output_shapes
:
�
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:���������
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"�      *
_output_shapes
:
�
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
w
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*(
_output_shapes
:����������
c
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*(
_output_shapes
:����������
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"�      *
_output_shapes
:
�
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
* 
_output_shapes
:
��
\
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2* 
_output_shapes
:
��
`
metrics/auc/zerosConst*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/true_positives
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:�
�
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
T0*
_output_shapes	
:�
o
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater* 
_output_shapes
:
��
o
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

DstT0*

SrcT0
* 
_output_shapes
:
��
c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*-
_class#
!loc:@metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_1Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/false_negatives
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
validate_shape(*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:�
�
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
_output_shapes	
:�
t
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot* 
_output_shapes
:
��
q
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_2Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/true_negatives
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:�
�
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
_output_shapes	
:�
z
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot* 
_output_shapes
:
��
q
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_3Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/false_positives
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:�
�
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
T0*
_output_shapes	
:�
w
metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater* 
_output_shapes
:
��
q
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*.
_class$
" loc:@metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:�
V
metrics/auc/add/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:�
�
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_2/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
T0*
_output_shapes	
:�
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
T0*
_output_shapes	
:�
�
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_4/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:�
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:�
Z
metrics/auc/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
T0*
_output_shapes	
:�
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:�
]
metrics/auc/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
|
metrics/auc/valueSummetrics/auc/Mulmetrics/auc/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
X
metrics/auc/add_6/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
T0*
_output_shapes	
:�
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:�
X
metrics/auc/add_8/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:�
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
T0*
_output_shapes	
:�
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:�
Y
metrics/auc/add_10/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:�
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
T0*
_output_shapes	
:�
\
metrics/auc/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
s
metrics/auc/truediv_1RealDivmetrics/auc/add_11metrics/auc/truediv_1/y*
T0*
_output_shapes	
:�
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:�
]
metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
�
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

*metrics/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*
_output_shapes
:

r
(metrics/softmax_cross_entropy_loss/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
�
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*$
_output_shapes
:
:

a
metrics/eval_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
�
initNoOp^global_step/Assign?^linear/text_ids_weighted_by_text_weights/weights/part_0/Assign!^linear/bias_weight/part_0/Assign^train_op/beta1_power/Assign^train_op/beta2_power/AssignD^linear/text_ids_weighted_by_text_weights/weights/part_0/Adam/AssignF^linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/Assign&^linear/bias_weight/part_0/Adam/Assign(^linear/bias_weight/part_0/Adam_1/Assign

init_1NoOp
"

group_depsNoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized7linear/text_ids_weighted_by_text_weights/weights/part_0*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedlinear/bias_weight/part_0*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedtrain_op/beta1_power*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedtrain_op/beta2_power*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedlinear/bias_weight/part_0/Adam*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized linear/bias_weight/part_0/Adam_1*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedmetrics/accuracy/total*
dtype0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedmetrics/accuracy/count*
dtype0*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedmetrics/auc/true_positives*
dtype0*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedmetrics/auc/false_negatives*
dtype0*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedmetrics/auc/true_negatives*
dtype0*-
_class#
!loc:@metrics/auc/true_negatives*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedmetrics/auc/false_positives*
dtype0*.
_class$
" loc:@metrics/auc/false_positives*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedmetrics/mean/total*
dtype0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedmetrics/mean/count*
dtype0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_16*
N*
T0
*
_output_shapes
:*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst*
dtype0*�
value�B�Bglobal_stepB7linear/text_ids_weighted_by_text_weights/weights/part_0Blinear/bias_weight/part_0Btrain_op/beta1_powerBtrain_op/beta2_powerB<linear/text_ids_weighted_by_text_weights/weights/part_0/AdamB>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1Blinear/bias_weight/part_0/AdamB linear/bias_weight/part_0/Adam_1Bmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/count*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
_output_shapes
:*
T0*
Tshape0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
g
$report_uninitialized_resources/ConstConst*
dtype0*
valueB *
_output_shapes
: 
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:���������*
T0
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized7linear/text_ids_weighted_by_text_weights/weights/part_0*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedlinear/bias_weight/part_0*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedtrain_op/beta1_power*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedtrain_op/beta2_power*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializedlinear/bias_weight/part_0/Adam*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitialized linear/bias_weight/part_0/Adam_1*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_8*
N	*
T0
*
_output_shapes
:	*

axis 
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:	
�
&report_uninitialized_variables_1/ConstConst*
dtype0*�
value�B�	Bglobal_stepB7linear/text_ids_weighted_by_text_weights/weights/part_0Blinear/bias_weight/part_0Btrain_op/beta1_powerBtrain_op/beta2_powerB<linear/text_ids_weighted_by_text_weights/weights/part_0/AdamB>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1Blinear/bias_weight/part_0/AdamB linear/bias_weight/part_0/Adam_1*
_output_shapes
:	
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:	*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:	*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
_output_shapes
:	*
T0*
Tshape0
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
_output_shapes
:	*
T0
*
Tshape0
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
�
init_2NoOp^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign

init_all_tablesNoOp
/
group_deps_1NoOp^init_2^init_all_tables
�
Merge/MergeSummaryMergeSummary7read_batch_features/file_name_queue/fraction_of_32_full/read_batch_features/fraction_over_10_of_10_full]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_fulltraining_loss/ScalarSummary*
_output_shapes
: *
N
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_2dd0cb1d0a6d4b0ab68c6b8c3c3f479b/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�	Bglobal_stepBlinear/bias_weightBlinear/bias_weight/AdamBlinear/bias_weight/Adam_1B0linear/text_ids_weighted_by_text_weights/weightsB5linear/text_ids_weighted_by_text_weights/weights/AdamB7linear/text_ids_weighted_by_text_weights/weights/Adam_1Btrain_op/beta1_powerBtrain_op/beta2_power*
_output_shapes
:	
�
save/SaveV2/shape_and_slicesConst*
dtype0*s
valuejBh	B B20 0,20B20 0,20B20 0,20B7179 20 0,7179:0,20B7179 20 0,7179:0,20B7179 20 0,7179:0,20B B *
_output_shapes
:	
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_steplinear/bias_weight/part_0/read#linear/bias_weight/part_0/Adam/read%linear/bias_weight/part_0/Adam_1/read<linear/text_ids_weighted_by_text_weights/weights/part_0/readAlinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readClinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readtrain_op/beta1_powertrain_op/beta2_power*
dtypes
2		
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*
_output_shapes
:*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2	*
_output_shapes
:
�
save/AssignAssignglobal_stepsave/RestoreV2*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
x
save/RestoreV2_1/tensor_namesConst*
dtype0*'
valueBBlinear/bias_weight*
_output_shapes
:
q
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueBB20 0,20*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assignlinear/bias_weight/part_0save/RestoreV2_1*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
}
save/RestoreV2_2/tensor_namesConst*
dtype0*,
value#B!Blinear/bias_weight/Adam*
_output_shapes
:
q
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueBB20 0,20*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assignlinear/bias_weight/part_0/Adamsave/RestoreV2_2*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_3/tensor_namesConst*
dtype0*.
value%B#Blinear/bias_weight/Adam_1*
_output_shapes
:
q
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueBB20 0,20*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assign linear/bias_weight/part_0/Adam_1save/RestoreV2_3*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
save/RestoreV2_4/tensor_namesConst*
dtype0*E
value<B:B0linear/text_ids_weighted_by_text_weights/weights*
_output_shapes
:
}
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*(
valueBB7179 20 0,7179:0,20*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assign7linear/text_ids_weighted_by_text_weights/weights/part_0save/RestoreV2_4*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
save/RestoreV2_5/tensor_namesConst*
dtype0*J
valueAB?B5linear/text_ids_weighted_by_text_weights/weights/Adam*
_output_shapes
:
}
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*(
valueBB7179 20 0,7179:0,20*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assign<linear/text_ids_weighted_by_text_weights/weights/part_0/Adamsave/RestoreV2_5*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
save/RestoreV2_6/tensor_namesConst*
dtype0*L
valueCBAB7linear/text_ids_weighted_by_text_weights/weights/Adam_1*
_output_shapes
:
}
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*(
valueBB7179 20 0,7179:0,20*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1save/RestoreV2_6*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
z
save/RestoreV2_7/tensor_namesConst*
dtype0*)
value BBtrain_op/beta1_power*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assigntrain_op/beta1_powersave/RestoreV2_7*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
z
save/RestoreV2_8/tensor_namesConst*
dtype0*)
value BBtrain_op/beta2_power*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assigntrain_op/beta2_powersave/RestoreV2_8*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard"�pE�     Q
_w	�F���@�AJ��
�@�@
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
p
	AssignSub
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint���������"
	containerstring "
shared_namestring �
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
�
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�
<
	LessEqual
x"T
y"T
z
"
Ttype:
2		
\
ListDiff
x"T
y"T
out"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
$

LogicalAnd
x

y

z
�


LogicalNot
x

y

:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
b
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParseExample

serialized	
names
sparse_keys*Nsparse

dense_keys*Ndense
dense_defaults2Tdense
sparse_indices	*Nsparse
sparse_values2sparse_types
sparse_shapes	*Nsparse
dense_values2Tdense"
Nsparseint("
Ndenseint("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
5
PreventGradient

input"T
output"T"	
Ttype
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
?
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( 
�
QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint���������
~
QueueDequeueV2

handle

components2component_types"!
component_types
list(type)(0"

timeout_msint���������
z
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint���������
v
QueueEnqueueV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint���������
#
QueueSizeV2

handle
size
Y
RandomShuffle

value"T
output"T"
seedint "
seed2int "	
Ttype�
�
RandomShuffleQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint���������"
min_after_dequeueint "
seedint "
seed2int "
	containerstring "
shared_namestring �
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
^
ReaderReadUpToV2
reader_handle
queue_handle
num_records	
keys

values
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
�

ScatterAdd
ref"T�
indices"Tindices
updates"T

output_ref"T�"
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
v

SegmentSum	
data"T
segment_ids"Tindices
output"T"
Ttype:
2	"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
y
SparseReorder
input_indices	
input_values"T
input_shape	
output_indices	
output_values"T"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
�
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
�
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
,
Sqrt
x"T
y"T"
Ttype:	
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
|
TFRecordReaderV2
reader_handle"
	containerstring "
shared_namestring "
compression_typestring �
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
�
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments
output"T"
Ttype:
2	"
Tindicestype:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirty��


global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
�
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
�
)read_batch_features/file_name_queue/inputConst*
dtype0*�
value�B�B/exout/features_train-00011-of-00014.tfrecord.gzB/exout/features_train-00013-of-00014.tfrecord.gzB/exout/features_train-00001-of-00014.tfrecord.gzB/exout/features_train-00006-of-00014.tfrecord.gzB/exout/features_train-00010-of-00014.tfrecord.gzB/exout/features_train-00008-of-00014.tfrecord.gzB/exout/features_train-00004-of-00014.tfrecord.gzB/exout/features_train-00005-of-00014.tfrecord.gzB/exout/features_train-00009-of-00014.tfrecord.gzB/exout/features_train-00012-of-00014.tfrecord.gzB/exout/features_train-00000-of-00014.tfrecord.gzB/exout/features_train-00003-of-00014.tfrecord.gzB/exout/features_train-00002-of-00014.tfrecord.gzB/exout/features_train-00007-of-00014.tfrecord.gz*
_output_shapes
:
j
(read_batch_features/file_name_queue/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
o
-read_batch_features/file_name_queue/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
�
+read_batch_features/file_name_queue/GreaterGreater(read_batch_features/file_name_queue/Size-read_batch_features/file_name_queue/Greater/y*
T0*
_output_shapes
: 
�
0read_batch_features/file_name_queue/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
�
8read_batch_features/file_name_queue/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
�
1read_batch_features/file_name_queue/Assert/AssertAssert+read_batch_features/file_name_queue/Greater8read_batch_features/file_name_queue/Assert/Assert/data_0*
	summarize*

T
2
�
,read_batch_features/file_name_queue/IdentityIdentity)read_batch_features/file_name_queue/input2^read_batch_features/file_name_queue/Assert/Assert*
T0*
_output_shapes
:
�
1read_batch_features/file_name_queue/RandomShuffleRandomShuffle,read_batch_features/file_name_queue/Identity*
seed2 *

seed *
T0*
_output_shapes
:
�
#read_batch_features/file_name_queueFIFOQueueV2*
capacity *
component_types
2*
_output_shapes
: *
shapes
: *
	container *
shared_name 
�
?read_batch_features/file_name_queue/file_name_queue_EnqueueManyQueueEnqueueManyV2#read_batch_features/file_name_queue1read_batch_features/file_name_queue/RandomShuffle*

timeout_ms���������*
Tcomponents
2
�
9read_batch_features/file_name_queue/file_name_queue_CloseQueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues( 
�
;read_batch_features/file_name_queue/file_name_queue_Close_1QueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues(
�
8read_batch_features/file_name_queue/file_name_queue_SizeQueueSizeV2#read_batch_features/file_name_queue*
_output_shapes
: 
�
(read_batch_features/file_name_queue/CastCast8read_batch_features/file_name_queue/file_name_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
n
)read_batch_features/file_name_queue/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
�
'read_batch_features/file_name_queue/mulMul(read_batch_features/file_name_queue/Cast)read_batch_features/file_name_queue/mul/y*
T0*
_output_shapes
: 
�
<read_batch_features/file_name_queue/fraction_of_32_full/tagsConst*
dtype0*H
value?B= B7read_batch_features/file_name_queue/fraction_of_32_full*
_output_shapes
: 
�
7read_batch_features/file_name_queue/fraction_of_32_fullScalarSummary<read_batch_features/file_name_queue/fraction_of_32_full/tags'read_batch_features/file_name_queue/mul*
T0*
_output_shapes
: 
�
)read_batch_features/read/TFRecordReaderV2TFRecordReaderV2*
shared_name *
	container *
compression_typeGZIP*
_output_shapes
: 
w
5read_batch_features/read/ReaderReadUpToV2/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
)read_batch_features/read/ReaderReadUpToV2ReaderReadUpToV2)read_batch_features/read/TFRecordReaderV2#read_batch_features/file_name_queue5read_batch_features/read/ReaderReadUpToV2/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_1TFRecordReaderV2*
shared_name *
	container *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_1/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_1ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_1#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_1/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_2TFRecordReaderV2*
shared_name *
	container *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_2/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_2ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_2#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_2/num_records*2
_output_shapes 
:���������:���������
�
+read_batch_features/read/TFRecordReaderV2_3TFRecordReaderV2*
shared_name *
	container *
compression_typeGZIP*
_output_shapes
: 
y
7read_batch_features/read/ReaderReadUpToV2_3/num_recordsConst*
dtype0	*
value	B	 R
*
_output_shapes
: 
�
+read_batch_features/read/ReaderReadUpToV2_3ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_3#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_3/num_records*2
_output_shapes 
:���������:���������
[
read_batch_features/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
(read_batch_features/random_shuffle_queueRandomShuffleQueueV2*
	container *
component_types
2*
_output_shapes
: *
min_after_dequeue
*
shapes
: : *
seed2 *

seed *
capacity*
shared_name 
�
read_batch_features/cond/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
q
!read_batch_features/cond/switch_tIdentity!read_batch_features/cond/Switch:1*
T0
*
_output_shapes
: 
o
!read_batch_features/cond/switch_fIdentityread_batch_features/cond/Switch*
T0
*
_output_shapes
: 
h
 read_batch_features/cond/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
@read_batch_features/cond/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue read_batch_features/cond/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1Switch)read_batch_features/read/ReaderReadUpToV2 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:���������:���������
�
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2Switch+read_batch_features/read/ReaderReadUpToV2:1 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:���������:���������
�
9read_batch_features/cond/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
+read_batch_features/cond/control_dependencyIdentity!read_batch_features/cond/switch_t:^read_batch_features/cond/random_shuffle_queue_EnqueueMany*4
_class*
(&loc:@read_batch_features/cond/switch_t*
T0
*
_output_shapes
: 
I
read_batch_features/cond/NoOpNoOp"^read_batch_features/cond/switch_f
�
-read_batch_features/cond/control_dependency_1Identity!read_batch_features/cond/switch_f^read_batch_features/cond/NoOp*4
_class*
(&loc:@read_batch_features/cond/switch_f*
T0
*
_output_shapes
: 
�
read_batch_features/cond/MergeMerge-read_batch_features/cond/control_dependency_1+read_batch_features/cond/control_dependency*
N*
T0
*
_output_shapes
: : 
�
!read_batch_features/cond_1/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_1/switch_tIdentity#read_batch_features/cond_1/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_1/switch_fIdentity!read_batch_features/cond_1/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_1/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_1/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_1:1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_1/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_1/control_dependencyIdentity#read_batch_features/cond_1/switch_t<^read_batch_features/cond_1/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_1/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_1/NoOpNoOp$^read_batch_features/cond_1/switch_f
�
/read_batch_features/cond_1/control_dependency_1Identity#read_batch_features/cond_1/switch_f ^read_batch_features/cond_1/NoOp*6
_class,
*(loc:@read_batch_features/cond_1/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_1/MergeMerge/read_batch_features/cond_1/control_dependency_1-read_batch_features/cond_1/control_dependency*
N*
T0
*
_output_shapes
: : 
�
!read_batch_features/cond_2/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_2/switch_tIdentity#read_batch_features/cond_2/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_2/switch_fIdentity!read_batch_features/cond_2/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_2/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_2/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_2"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_2:1"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_2/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_2/control_dependencyIdentity#read_batch_features/cond_2/switch_t<^read_batch_features/cond_2/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_2/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_2/NoOpNoOp$^read_batch_features/cond_2/switch_f
�
/read_batch_features/cond_2/control_dependency_1Identity#read_batch_features/cond_2/switch_f ^read_batch_features/cond_2/NoOp*6
_class,
*(loc:@read_batch_features/cond_2/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_2/MergeMerge/read_batch_features/cond_2/control_dependency_1-read_batch_features/cond_2/control_dependency*
N*
T0
*
_output_shapes
: : 
�
!read_batch_features/cond_3/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
u
#read_batch_features/cond_3/switch_tIdentity#read_batch_features/cond_3/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_3/switch_fIdentity!read_batch_features/cond_3/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_3/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
�
Bread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/SwitchSwitch(read_batch_features/random_shuffle_queue"read_batch_features/cond_3/pred_id*;
_class1
/-loc:@read_batch_features/random_shuffle_queue*
T0*
_output_shapes
: : 
�
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_3"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:���������:���������
�
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_3:1"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:���������:���������
�
;read_batch_features/cond_3/random_shuffle_queue_EnqueueManyQueueEnqueueManyV2Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2:1*

timeout_ms���������*
Tcomponents
2
�
-read_batch_features/cond_3/control_dependencyIdentity#read_batch_features/cond_3/switch_t<^read_batch_features/cond_3/random_shuffle_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_3/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_3/NoOpNoOp$^read_batch_features/cond_3/switch_f
�
/read_batch_features/cond_3/control_dependency_1Identity#read_batch_features/cond_3/switch_f ^read_batch_features/cond_3/NoOp*6
_class,
*(loc:@read_batch_features/cond_3/switch_f*
T0
*
_output_shapes
: 
�
 read_batch_features/cond_3/MergeMerge/read_batch_features/cond_3/control_dependency_1-read_batch_features/cond_3/control_dependency*
N*
T0
*
_output_shapes
: : 
�
.read_batch_features/random_shuffle_queue_CloseQueueCloseV2(read_batch_features/random_shuffle_queue*
cancel_pending_enqueues( 
�
0read_batch_features/random_shuffle_queue_Close_1QueueCloseV2(read_batch_features/random_shuffle_queue*
cancel_pending_enqueues(
~
-read_batch_features/random_shuffle_queue_SizeQueueSizeV2(read_batch_features/random_shuffle_queue*
_output_shapes
: 
[
read_batch_features/sub/yConst*
dtype0*
value	B :
*
_output_shapes
: 
�
read_batch_features/subSub-read_batch_features/random_shuffle_queue_Sizeread_batch_features/sub/y*
T0*
_output_shapes
: 
_
read_batch_features/Maximum/xConst*
dtype0*
value	B : *
_output_shapes
: 

read_batch_features/MaximumMaximumread_batch_features/Maximum/xread_batch_features/sub*
T0*
_output_shapes
: 
m
read_batch_features/CastCastread_batch_features/Maximum*

DstT0*

SrcT0*
_output_shapes
: 
^
read_batch_features/mul/yConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
t
read_batch_features/mulMulread_batch_features/Castread_batch_features/mul/y*
T0*
_output_shapes
: 
�
4read_batch_features/fraction_over_10_of_10_full/tagsConst*
dtype0*@
value7B5 B/read_batch_features/fraction_over_10_of_10_full*
_output_shapes
: 
�
/read_batch_features/fraction_over_10_of_10_fullScalarSummary4read_batch_features/fraction_over_10_of_10_full/tagsread_batch_features/mul*
T0*
_output_shapes
: 
W
read_batch_features/nConst*
dtype0*
value	B :
*
_output_shapes
: 
�
read_batch_featuresQueueDequeueManyV2(read_batch_features/random_shuffle_queueread_batch_features/n*

timeout_ms���������*
component_types
2* 
_output_shapes
:
:

j
(read_batch_features/ParseExample/key_keyConst*
dtype0	*
value	B	 R *
_output_shapes
: 
q
.read_batch_features/ParseExample/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
(read_batch_features/ParseExample/ReshapeReshape(read_batch_features/ParseExample/key_key.read_batch_features/ParseExample/Reshape/shape*
Tshape0*
T0	*
_output_shapes
: 
i
&read_batch_features/ParseExample/ConstConst*
dtype0	*
valueB	 *
_output_shapes
: 
v
3read_batch_features/ParseExample/ParseExample/namesConst*
dtype0*
valueB *
_output_shapes
: 
�
;read_batch_features/ParseExample/ParseExample/sparse_keys_0Const*
dtype0*
valueB Btext_ids*
_output_shapes
: 
�
;read_batch_features/ParseExample/ParseExample/sparse_keys_1Const*
dtype0*
valueB Btext_weights*
_output_shapes
: 
~
:read_batch_features/ParseExample/ParseExample/dense_keys_0Const*
dtype0*
valueB	 Bkey*
_output_shapes
: 
�
:read_batch_features/ParseExample/ParseExample/dense_keys_1Const*
dtype0*
valueB Btarget*
_output_shapes
: 
�
-read_batch_features/ParseExample/ParseExampleParseExampleread_batch_features:13read_batch_features/ParseExample/ParseExample/names;read_batch_features/ParseExample/ParseExample/sparse_keys_0;read_batch_features/ParseExample/ParseExample/sparse_keys_1:read_batch_features/ParseExample/ParseExample/dense_keys_0:read_batch_features/ParseExample/ParseExample/dense_keys_1(read_batch_features/ParseExample/Reshape&read_batch_features/ParseExample/Const*
dense_shapes
: : *p
_output_shapes^
\:���������:���������:���������:���������:::
:
*
Ndense*
sparse_types
2	*
Tdense
2		*
Nsparse
�
read_batch_features/fifo_queueFIFOQueueV2*
capacityd* 
component_types
2								*
_output_shapes
: *
shapes
 *
	container *
shared_name 
j
#read_batch_features/fifo_queue_SizeQueueSizeV2read_batch_features/fifo_queue*
_output_shapes
: 
w
read_batch_features/Cast_1Cast#read_batch_features/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
`
read_batch_features/mul_1/yConst*
dtype0*
valueB
 *
�#<*
_output_shapes
: 
z
read_batch_features/mul_1Mulread_batch_features/Cast_1read_batch_features/mul_1/y*
T0*
_output_shapes
: 
�
bread_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full/tagsConst*
dtype0*n
valueeBc B]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full*
_output_shapes
: 
�
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_fullScalarSummarybread_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full/tagsread_batch_features/mul_1*
T0*
_output_shapes
: 
�
&read_batch_features/fifo_queue_enqueueQueueEnqueueV2read_batch_features/fifo_queue/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms���������*
Tcomponents
2								
�
(read_batch_features/fifo_queue_enqueue_1QueueEnqueueV2read_batch_features/fifo_queue/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms���������*
Tcomponents
2								
s
$read_batch_features/fifo_queue_CloseQueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues( 
u
&read_batch_features/fifo_queue_Close_1QueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues(
�
&read_batch_features/fifo_queue_DequeueQueueDequeueV2read_batch_features/fifo_queue*

timeout_ms���������* 
component_types
2								*v
_output_shapesd
b:
:
:���������:���������::���������:���������::

Y
ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�

ExpandDims
ExpandDims&read_batch_features/fifo_queue_DequeueExpandDims/dim*

Tdim0*
T0	*
_output_shapes

:

[
ExpandDims_1/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
ExpandDims_1
ExpandDims(read_batch_features/fifo_queue_Dequeue:1ExpandDims_1/dim*

Tdim0*
T0	*
_output_shapes

:

V
linear/linear/mod/yConst*
dtype0	*
value
B	 R�8*
_output_shapes
: 
�
linear/linear/modFloorMod(read_batch_features/fifo_queue_Dequeue:3linear/linear/mod/y*
T0	*#
_output_shapes
:���������
�
Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/ConstConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB	�8*    *
_output_shapes
:	�8
�
7linear/text_ids_weighted_by_text_weights/weights/part_0
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign7linear/text_ids_weighted_by_text_weights/weights/part_0Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/Const*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
<linear/text_ids_weighted_by_text_weights/weights/part_0/readIdentity7linear/text_ids_weighted_by_text_weights/weights/part_0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SliceSlice(read_batch_features/fifo_queue_Dequeue:4elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/begindlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ProdProd_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
�
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GatherGather(read_batch_features/fifo_queue_Dequeue:4hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
�
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shapePack^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Prod`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather*
_output_shapes
:*

axis *
T0	*
N
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeSparseReshape(read_batch_features/fifo_queue_Dequeue:2(read_batch_features/fifo_queue_Dequeue:4qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shape*-
_output_shapes
:���������:
�
plinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/IdentityIdentitylinear/linear/mod*
T0	*#
_output_shapes
:���������
�
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualGreaterEqualplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityhlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/y*
T0	*#
_output_shapes
:���������
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterGreater(read_batch_features/fifo_queue_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/y*
T0*#
_output_shapes
:���������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd
LogicalAndflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater*#
_output_shapes
:���������
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/WhereWheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:���������
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ReshapeReshape_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Whereglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2Gatherplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityIdentityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1Wheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:���������
�
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1Reshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shape*
Tshape0*
T0	*#
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:���������
�
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4Gather(read_batch_features/fifo_queue_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Identityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_sliceStridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/CastCast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/rangeRangeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startrlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Castylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:���������
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1Castslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffListDifftlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims
ExpandDims}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseSparseToDensevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_values�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ReshapeReshapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:���������
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_like	ZerosLikeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:���������
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concatConcatV2ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_likeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axis*'
_output_shapes
:���������*

Tidx0*
T0	*
N
�
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ShapeShapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
�
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/FillFillslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Shapeslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:���������*

Tidx0*
T0	*
N
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Fill{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:���������*

Tidx0*
T0	*
N
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorderSparseReordervlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*6
_output_shapes$
":���������:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/IdentityIdentityblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*
_output_shapes
:
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_sliceStridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/CastCast}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/rangeRange{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/starttlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/delta*

Tidx0*#
_output_shapes
:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1Castulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range*

DstT0	*

SrcT0*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffListDiffvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2StridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
�
~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims
ExpandDimslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDenseSparseToDensexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffzlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_values�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:���������
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ReshapeReshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:���������
�
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like	ZerosLikewlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape*
T0	*'
_output_shapes
:���������
�
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshapezlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axis*'
_output_shapes
:���������*

Tidx0*
T0	*
N
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ShapeShapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff*
out_type0*
T0	*
_output_shapes
:
�
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/FillFillulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Shapeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Const*
T0*#
_output_shapes
:���������
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axis*'
_output_shapes
:���������*

Tidx0*
T0	*
N
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axis*#
_output_shapes
:���������*

Tidx0*
T0*
N
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorderSparseReorderxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0*6
_output_shapes$
":���������:���������
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/IdentityIdentitydlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0	*
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_sliceStridedSlice{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/CastCastlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:���������
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupGather<linear/text_ids_weighted_by_text_weights/weights/part_0/read}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
validate_indices(*
Tparams0*
Tindices0	*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*'
_output_shapes
:���������
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/RankConst*
dtype0*
value	B :*
_output_shapes
: 
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/subSubvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Rankwlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/y*
T0*
_output_shapes
: 
�
�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims
ExpandDimsulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
�
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
�
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/FillFill|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/value*
T0*#
_output_shapes
:���������
�
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ShapeShapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1*
out_type0*
T0*
_output_shapes
:
�
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Shapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axis*#
_output_shapes
:���������*

Tidx0*
T0*
N
�
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ReshapeReshapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat*
Tshape0*
T0*'
_output_shapes
:���������
�
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulMul�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:���������
�
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse
SegmentSumulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:���������
�
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2Reshape{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shape*
Tshape0*
T0
*'
_output_shapes
:���������
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ShapeShapeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
�
mlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_sliceStridedSlice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shapemlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackolinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
�
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stackPackalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice*
_output_shapes
:*

axis *
T0*
N
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/TileTileclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:������������������
�
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like	ZerosLikeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
Ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weightsSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tiledlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_likeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/CastCast(read_batch_features/fifo_queue_Dequeue:4*

DstT0*

SrcT0	*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1Slice^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Castglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1ShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
�
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
�
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/sizeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2Slicealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concatConcatV2alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3ReshapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat*
Tshape0*
T0*'
_output_shapes
:���������
l
linear/linear/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
linear/linear/ReshapeReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3linear/linear/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:���������
�
+linear/bias_weight/part_0/Initializer/ConstConst*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
valueB*    *
_output_shapes
:
�
linear/bias_weight/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
 linear/bias_weight/part_0/AssignAssignlinear/bias_weight/part_0+linear/bias_weight/part_0/Initializer/Const*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
linear/bias_weight/part_0/readIdentitylinear/bias_weight/part_0*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
c
linear/bias_weightIdentitylinear/bias_weight/part_0/read*
T0*
_output_shapes
:
�
linear/linear/BiasAddBiasAddlinear/linear/Reshapelinear/bias_weight*
data_formatNHWC*
T0*'
_output_shapes
:���������
m
predictions/probabilitiesSoftmaxlinear/linear/BiasAdd*
T0*'
_output_shapes
:���������
_
predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
predictions/classesArgMaxlinear/linear/BiasAddpredictions/classes/dimension*

Tidx0*
T0*#
_output_shapes
:���������
�
0training_loss/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*
_output_shapes
:

x
.training_loss/softmax_cross_entropy_loss/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
�
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*$
_output_shapes
:
:

]
training_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
[
train_op/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
]
train_op/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
t
train_op/gradients/FillFilltrain_op/gradients/Shapetrain_op/gradients/Const*
T0*
_output_shapes
: 
}
3train_op/gradients/training_loss_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
-train_op/gradients/training_loss_grad/ReshapeReshapetrain_op/gradients/Fill3train_op/gradients/training_loss_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
~
4train_op/gradients/training_loss_grad/Tile/multiplesConst*
dtype0*
valueB:
*
_output_shapes
:
�
*train_op/gradients/training_loss_grad/TileTile-train_op/gradients/training_loss_grad/Reshape4train_op/gradients/training_loss_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:

u
+train_op/gradients/training_loss_grad/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
p
-train_op/gradients/training_loss_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
u
+train_op/gradients/training_loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
*train_op/gradients/training_loss_grad/ProdProd+train_op/gradients/training_loss_grad/Shape+train_op/gradients/training_loss_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
w
-train_op/gradients/training_loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
,train_op/gradients/training_loss_grad/Prod_1Prod-train_op/gradients/training_loss_grad/Shape_1-train_op/gradients/training_loss_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
q
/train_op/gradients/training_loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
-train_op/gradients/training_loss_grad/MaximumMaximum,train_op/gradients/training_loss_grad/Prod_1/train_op/gradients/training_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
.train_op/gradients/training_loss_grad/floordivFloorDiv*train_op/gradients/training_loss_grad/Prod-train_op/gradients/training_loss_grad/Maximum*
T0*
_output_shapes
: 
�
*train_op/gradients/training_loss_grad/CastCast.train_op/gradients/training_loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
-train_op/gradients/training_loss_grad/truedivRealDiv*train_op/gradients/training_loss_grad/Tile*train_op/gradients/training_loss_grad/Cast*
T0*
_output_shapes
:


train_op/gradients/zeros_like	ZerosLike*training_loss/softmax_cross_entropy_loss:1*
T0*
_output_shapes

:

�
Ptrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/PreventGradientPreventGradient*training_loss/softmax_cross_entropy_loss:1*
T0*
_output_shapes

:

�
Otrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
Ktrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims
ExpandDims-train_op/gradients/training_loss_grad/truedivOtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:

�
Dtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mulMulKtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/ExpandDimsPtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/PreventGradient*
T0*
_output_shapes

:

�
9train_op/gradients/linear/linear/BiasAdd_grad/BiasAddGradBiasAddGradDtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mul*
data_formatNHWC*
T0*
_output_shapes
:
�
3train_op/gradients/linear/linear/Reshape_grad/ShapeShapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3*
out_type0*
T0*
_output_shapes
:
�
5train_op/gradients/linear/linear/Reshape_grad/ReshapeReshapeDtrain_op/gradients/training_loss/softmax_cross_entropy_loss_grad/mul3train_op/gradients/linear/linear/Reshape_grad/Shape*
Tshape0*
T0*
_output_shapes

:

�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/ShapeShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/ReshapeReshape5train_op/gradients/linear/linear/Reshape_grad/Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Shape*
Tshape0*
T0*
_output_shapes

:

�
|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like	ZerosLikedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like*
T0*'
_output_shapes
:���������
�
xtrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/SelectSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tile�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Reshape|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like*
T0*
_output_shapes

:

�
ztrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/Select_1Select^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tile|train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/zeros_like�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3_grad/Reshape*
T0*
_output_shapes

:

�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/GatherGatherztrain_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights_grad/Select_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
validate_indices(*
Tparams0*
Tindices0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/ShapeShape�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1Shapeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
out_type0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgsBroadcastGradientArgs�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mulMul�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/Gatherylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/SumSum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/ReshapeReshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul_1Mul�linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse_grad/Gather*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum_1Sum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/mul_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Reshape_1Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Sum_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ShapeConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB"     *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/SizeSize}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
out_type0*
T0	*
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims
ExpandDims�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Size�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_sliceStridedSlice�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Shape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concatConcatV2�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/strided_slice�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ReshapeReshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mul_grad/Reshape�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/concat*
Tshape0*
T0*0
_output_shapes
:������������������
�
�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape_1Reshape}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ExpandDims*
Tshape0*
T0	*#
_output_shapes
:���������
�
"train_op/beta1_power/initial_valueConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *fff?*
_output_shapes
: 
�
train_op/beta1_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
train_op/beta1_power/AssignAssigntrain_op/beta1_power"train_op/beta1_power/initial_value*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
train_op/beta1_power/readIdentitytrain_op/beta1_power*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
"train_op/beta2_power/initial_valueConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *w�?*
_output_shapes
: 
�
train_op/beta2_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
train_op/beta2_power/AssignAssigntrain_op/beta2_power"train_op/beta2_power/initial_value*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
train_op/beta2_power/readIdentitytrain_op/beta2_power*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
e
train_op/zerosConst*
dtype0*
valueB	�8*    *
_output_shapes
:	�8
�
<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/AssignAssign<linear/text_ids_weighted_by_text_weights/weights/part_0/Adamtrain_op/zeros*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
Alinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readIdentity<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
g
train_op/zeros_1Const*
dtype0*
valueB	�8*    *
_output_shapes
:	�8
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1
VariableV2*
	container *
_output_shapes
:	�8*
dtype0*
shape:	�8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
�
Elinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/AssignAssign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1train_op/zeros_1*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readIdentity>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
]
train_op/zeros_2Const*
dtype0*
valueB*    *
_output_shapes
:
�
linear/bias_weight/part_0/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
%linear/bias_weight/part_0/Adam/AssignAssignlinear/bias_weight/part_0/Adamtrain_op/zeros_2*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
#linear/bias_weight/part_0/Adam/readIdentitylinear/bias_weight/part_0/Adam*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
]
train_op/zeros_3Const*
dtype0*
valueB*    *
_output_shapes
:
�
 linear/bias_weight/part_0/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*,
_class"
 loc:@linear/bias_weight/part_0*
shared_name 
�
'linear/bias_weight/part_0/Adam_1/AssignAssign linear/bias_weight/part_0/Adam_1train_op/zeros_3*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
%linear/bias_weight/part_0/Adam_1/readIdentity linear/bias_weight/part_0/Adam_1*,
_class"
 loc:@linear/bias_weight/part_0*
T0*
_output_shapes
:
`
train_op/Adam/learning_rateConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
X
train_op/Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
X
train_op/Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Z
train_op/Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueUnique�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ShapeShapeStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Unique*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
out_type0*
T0	*
_output_shapes
:
�
`train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stackConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB: *
_output_shapes
:
�
btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_1Const*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB:*
_output_shapes
:
�
btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_2Const*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB:*
_output_shapes
:
�
Ztrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_sliceStridedSliceRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Shape`train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stackbtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_1btrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSumUnsortedSegmentSum�train_op/gradients/linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookup_grad/ReshapeUtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Unique:1Ztrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/strided_slice*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
Tindices0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/subSubRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub/xtrain_op/beta2_power/read*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Qtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/SqrtSqrtPtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mulMultrain_op/Adam/learning_rateQtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1/xtrain_op/beta1_power/read*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truedivRealDivPtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mulRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2/xtrain_op/Adam/beta1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_1Mul_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSumRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_2MulAlinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readtrain_op/Adam/beta1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign<linear/text_ids_weighted_by_text_weights/weights/part_0/AdamRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_2*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Wtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd
ScatterAddStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
Tindices0	*
use_locking( *
T0*
_output_shapes
:	�8
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_3Mul_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSum_train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UnsortedSegmentSum*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Ttrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3/xConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB
 *  �?*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3SubTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3/xtrain_op/Adam/beta2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_4MulRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_3Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/sub_3*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*0
_output_shapes
:������������������
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_5MulClinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readtrain_op/Adam/beta2*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Utrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Assign_1Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_5*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Ytrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1
ScatterAddUtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Assign_1Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/UniqueRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_4*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
Tindices0	*
use_locking( *
T0*
_output_shapes
:	�8
�
Strain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt_1SqrtYtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Rtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_6MulTtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truedivWtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/addAddStrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/Sqrt_1train_op/Adam/epsilon*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truediv_1RealDivRtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/mul_6Ptrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/add*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	�8
�
Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignSub	AssignSub7linear/text_ids_weighted_by_text_weights/weights/part_0Vtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/truediv_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
:	�8
�
Wtrain_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_depsNoOpW^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/AssignSubX^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAddZ^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/ScatterAdd_1*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0
�
8train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam	ApplyAdamlinear/bias_weight/part_0linear/bias_weight/part_0/Adam linear/bias_weight/part_0/Adam_1train_op/beta1_power/readtrain_op/beta2_power/readtrain_op/Adam/learning_ratetrain_op/Adam/beta1train_op/Adam/beta2train_op/Adam/epsilon9train_op/gradients/linear/linear/BiasAdd_grad/BiasAddGrad*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking( *
T0*
_output_shapes
:
�
train_op/Adam/mulMultrain_op/beta1_power/readtrain_op/Adam/beta1X^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
train_op/Adam/AssignAssigntrain_op/beta1_powertrain_op/Adam/mul*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
: 
�
train_op/Adam/mul_1Multrain_op/beta2_power/readtrain_op/Adam/beta2X^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
: 
�
train_op/Adam/Assign_1Assigntrain_op/beta2_powertrain_op/Adam/mul_1*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking( *
T0*
_output_shapes
: 
�
train_op/Adam/updateNoOpX^train_op/Adam/update_linear/text_ids_weighted_by_text_weights/weights/part_0/group_deps9^train_op/Adam/update_linear/bias_weight/part_0/ApplyAdam^train_op/Adam/Assign^train_op/Adam/Assign_1
�
train_op/Adam/valueConst^train_op/Adam/update*
dtype0	*
_class
loc:@global_step*
value	B	 R*
_output_shapes
: 
�
train_op/Adam	AssignAddglobal_steptrain_op/Adam/value*
_class
loc:@global_step*
use_locking( *
T0	*
_output_shapes
: 
�
,metrics/remove_squeezable_dimensions/SqueezeSqueezeExpandDims_1*
squeeze_dims

���������*
T0	*
_output_shapes
:

~
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*
_output_shapes
:

Z
metrics/ToFloatCastmetrics/Equal*

DstT0*

SrcT0
*
_output_shapes
:

[
metrics/accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
validate_shape(*)
_class
loc:@metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
�
metrics/accuracy/total/readIdentitymetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
T0*
_output_shapes
: 
]
metrics/accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
validate_shape(*)
_class
loc:@metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
�
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
T0*
_output_shapes
: 
W
metrics/accuracy/SizeConst*
dtype0*
value	B :
*
_output_shapes
: 
i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
`
metrics/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*)
_class
loc:@metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/accuracy/AssignAdd_1	AssignAddmetrics/accuracy/countmetrics/accuracy/ToFloat_1*)
_class
loc:@metrics/accuracy/count*
use_locking( *
T0*
_output_shapes
: 
_
metrics/accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
}
metrics/accuracy/GreaterGreatermetrics/accuracy/count/readmetrics/accuracy/Greater/y*
T0*
_output_shapes
: 
~
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
T0*
_output_shapes
: 
]
metrics/accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
T0*
_output_shapes
: 
a
metrics/accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
�
metrics/accuracy/truediv_1RealDivmetrics/accuracy/AssignAddmetrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
a
metrics/accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
N
metrics/RankConst*
dtype0*
value	B :*
_output_shapes
: 
U
metrics/LessEqual/yConst*
dtype0*
value	B :*
_output_shapes
: 
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
T0*
_output_shapes
: 
�
metrics/Assert/ConstConst*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
�
metrics/Assert/Assert/data_0Const*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*
	summarize*

T
2
�
metrics/Reshape/shapeConst^metrics/Assert/Assert*
dtype0*
valueB:
���������*
_output_shapes
:
r
metrics/ReshapeReshapeExpandDims_1metrics/Reshape/shape*
Tshape0*
T0	*
_output_shapes
:

]
metrics/one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
^
metrics/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
W
metrics/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
TI0	*
_output_shapes

:
*
T0*
axis���������
]
metrics/CastCastmetrics/one_hot*

DstT0
*

SrcT0*
_output_shapes

:

j
metrics/auc/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:���������
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"   ����*
_output_shapes
:
�
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:	�
d
metrics/auc/ShapeShapemetrics/auc/Reshape*
out_type0*
T0*
_output_shapes
:
i
metrics/auc/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
k
!metrics/auc/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
k
!metrics/auc/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_sliceStridedSlicemetrics/auc/Shapemetrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
�
metrics/auc/ConstConst*
dtype0*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
_output_shapes	
:�
d
metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	�
U
metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
_output_shapes
:*

axis *
T0*
N
�
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:����������
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
T0*
_output_shapes
: 
]
metrics/auc/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
T0*
_output_shapes
:
�
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:���������
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"�      *
_output_shapes
:
�
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:����������
w
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*(
_output_shapes
:����������
c
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*(
_output_shapes
:����������
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"�      *
_output_shapes
:
�
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
* 
_output_shapes
:
��
\
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2* 
_output_shapes
:
��
`
metrics/auc/zerosConst*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/true_positives
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:�
�
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
T0*
_output_shapes	
:�
o
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater* 
_output_shapes
:
��
o
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

DstT0*

SrcT0
* 
_output_shapes
:
��
c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:�
�
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*-
_class#
!loc:@metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_1Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/false_negatives
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
validate_shape(*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:�
�
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
_output_shapes	
:�
t
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot* 
_output_shapes
:
��
q
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:�
�
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_2Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/true_negatives
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:�
�
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
_output_shapes	
:�
z
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot* 
_output_shapes
:
��
q
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:�
�
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:�
b
metrics/auc/zeros_3Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
metrics/auc/false_positives
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:�
�
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
T0*
_output_shapes	
:�
w
metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater* 
_output_shapes
:
��
q
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

DstT0*

SrcT0
* 
_output_shapes
:
��
e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
�
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:�
�
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*.
_class$
" loc:@metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:�
V
metrics/auc/add/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:�
�
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_2/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
T0*
_output_shapes	
:�
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
T0*
_output_shapes	
:�
�
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_4/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:�
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:�
Z
metrics/auc/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
T0*
_output_shapes	
:�
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:�
]
metrics/auc/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
|
metrics/auc/valueSummetrics/auc/Mulmetrics/auc/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
X
metrics/auc/add_6/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
T0*
_output_shapes	
:�
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:�
X
metrics/auc/add_8/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:�
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
T0*
_output_shapes	
:�
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:�
Y
metrics/auc/add_10/yConst*
dtype0*
valueB
 *�7�5*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:�
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:�*
_output_shapes
:
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:�*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
T0*
_output_shapes	
:�
\
metrics/auc/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
s
metrics/auc/truediv_1RealDivmetrics/auc/add_11metrics/auc/truediv_1/y*
T0*
_output_shapes	
:�
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:�
]
metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
�
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

*metrics/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*
_output_shapes
:

r
(metrics/softmax_cross_entropy_loss/ShapeConst*
dtype0*
valueB:
*
_output_shapes
:
�
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*$
_output_shapes
:
:

a
metrics/eval_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
�
initNoOp^global_step/Assign?^linear/text_ids_weighted_by_text_weights/weights/part_0/Assign!^linear/bias_weight/part_0/Assign^train_op/beta1_power/Assign^train_op/beta2_power/AssignD^linear/text_ids_weighted_by_text_weights/weights/part_0/Adam/AssignF^linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/Assign&^linear/bias_weight/part_0/Adam/Assign(^linear/bias_weight/part_0/Adam_1/Assign

init_1NoOp
"

group_depsNoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized7linear/text_ids_weighted_by_text_weights/weights/part_0*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedlinear/bias_weight/part_0*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedtrain_op/beta1_power*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedtrain_op/beta2_power*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedlinear/bias_weight/part_0/Adam*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized linear/bias_weight/part_0/Adam_1*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedmetrics/accuracy/total*
dtype0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedmetrics/accuracy/count*
dtype0*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedmetrics/auc/true_positives*
dtype0*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedmetrics/auc/false_negatives*
dtype0*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedmetrics/auc/true_negatives*
dtype0*-
_class#
!loc:@metrics/auc/true_negatives*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedmetrics/auc/false_positives*
dtype0*.
_class$
" loc:@metrics/auc/false_positives*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedmetrics/mean/total*
dtype0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedmetrics/mean/count*
dtype0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_16*
_output_shapes
:*

axis *
T0
*
N
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst*
dtype0*�
value�B�Bglobal_stepB7linear/text_ids_weighted_by_text_weights/weights/part_0Blinear/bias_weight/part_0Btrain_op/beta1_powerBtrain_op/beta2_powerB<linear/text_ids_weighted_by_text_weights/weights/part_0/AdamB>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1Blinear/bias_weight/part_0/AdamB linear/bias_weight/part_0/Adam_1Bmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/count*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
_output_shapes
:*

axis *
T0*
N
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
g
$report_uninitialized_resources/ConstConst*
dtype0*
valueB *
_output_shapes
: 
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*#
_output_shapes
:���������*

Tidx0*
T0*
N
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized7linear/text_ids_weighted_by_text_weights/weights/part_0*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedlinear/bias_weight/part_0*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedtrain_op/beta1_power*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedtrain_op/beta2_power*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized<linear/text_ids_weighted_by_text_weights/weights/part_0/Adam*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializedlinear/bias_weight/part_0/Adam*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitialized linear/bias_weight/part_0/Adam_1*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_8*
_output_shapes
:	*

axis *
T0
*
N	
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:	
�
&report_uninitialized_variables_1/ConstConst*
dtype0*�
value�B�	Bglobal_stepB7linear/text_ids_weighted_by_text_weights/weights/part_0Blinear/bias_weight/part_0Btrain_op/beta1_powerBtrain_op/beta2_powerB<linear/text_ids_weighted_by_text_weights/weights/part_0/AdamB>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1Blinear/bias_weight/part_0/AdamB linear/bias_weight/part_0/Adam_1*
_output_shapes
:	
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:	*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:	*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
_output_shapes
:*

axis *
T0*
N
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:	
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:	
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
�
init_2NoOp^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign

init_all_tablesNoOp
/
group_deps_1NoOp^init_2^init_all_tables
�
Merge/MergeSummaryMergeSummary7read_batch_features/file_name_queue/fraction_of_32_full/read_batch_features/fraction_over_10_of_10_full]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_fulltraining_loss/ScalarSummary*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_2dd0cb1d0a6d4b0ab68c6b8c3c3f479b/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�	Bglobal_stepBlinear/bias_weightBlinear/bias_weight/AdamBlinear/bias_weight/Adam_1B0linear/text_ids_weighted_by_text_weights/weightsB5linear/text_ids_weighted_by_text_weights/weights/AdamB7linear/text_ids_weighted_by_text_weights/weights/Adam_1Btrain_op/beta1_powerBtrain_op/beta2_power*
_output_shapes
:	
�
save/SaveV2/shape_and_slicesConst*
dtype0*s
valuejBh	B B20 0,20B20 0,20B20 0,20B7179 20 0,7179:0,20B7179 20 0,7179:0,20B7179 20 0,7179:0,20B B *
_output_shapes
:	
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_steplinear/bias_weight/part_0/read#linear/bias_weight/part_0/Adam/read%linear/bias_weight/part_0/Adam_1/read<linear/text_ids_weighted_by_text_weights/weights/part_0/readAlinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/readClinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/readtrain_op/beta1_powertrain_op/beta2_power*
dtypes
2		
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*

axis *
T0*
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2	*
_output_shapes
:
�
save/AssignAssignglobal_stepsave/RestoreV2*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
x
save/RestoreV2_1/tensor_namesConst*
dtype0*'
valueBBlinear/bias_weight*
_output_shapes
:
q
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueBB20 0,20*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assignlinear/bias_weight/part_0save/RestoreV2_1*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
}
save/RestoreV2_2/tensor_namesConst*
dtype0*,
value#B!Blinear/bias_weight/Adam*
_output_shapes
:
q
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueBB20 0,20*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assignlinear/bias_weight/part_0/Adamsave/RestoreV2_2*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_3/tensor_namesConst*
dtype0*.
value%B#Blinear/bias_weight/Adam_1*
_output_shapes
:
q
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueBB20 0,20*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assign linear/bias_weight/part_0/Adam_1save/RestoreV2_3*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
�
save/RestoreV2_4/tensor_namesConst*
dtype0*E
value<B:B0linear/text_ids_weighted_by_text_weights/weights*
_output_shapes
:
}
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*(
valueBB7179 20 0,7179:0,20*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assign7linear/text_ids_weighted_by_text_weights/weights/part_0save/RestoreV2_4*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
save/RestoreV2_5/tensor_namesConst*
dtype0*J
valueAB?B5linear/text_ids_weighted_by_text_weights/weights/Adam*
_output_shapes
:
}
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*(
valueBB7179 20 0,7179:0,20*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assign<linear/text_ids_weighted_by_text_weights/weights/part_0/Adamsave/RestoreV2_5*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
�
save/RestoreV2_6/tensor_namesConst*
dtype0*L
valueCBAB7linear/text_ids_weighted_by_text_weights/weights/Adam_1*
_output_shapes
:
}
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*(
valueBB7179 20 0,7179:0,20*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1save/RestoreV2_6*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	�8
z
save/RestoreV2_7/tensor_namesConst*
dtype0*)
value BBtrain_op/beta1_power*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assigntrain_op/beta1_powersave/RestoreV2_7*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
z
save/RestoreV2_8/tensor_namesConst*
dtype0*)
value BBtrain_op/beta2_power*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assigntrain_op/beta2_powersave/RestoreV2_8*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"�%
cond_context�%�%
�
"read_batch_features/cond/cond_text"read_batch_features/cond/pred_id:0#read_batch_features/cond/switch_t:0 *�
-read_batch_features/cond/control_dependency:0
"read_batch_features/cond/pred_id:0
Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch:1
Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1:1
Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2:1
#read_batch_features/cond/switch_t:0
*read_batch_features/random_shuffle_queue:0
+read_batch_features/read/ReaderReadUpToV2:0
+read_batch_features/read/ReaderReadUpToV2:1s
+read_batch_features/read/ReaderReadUpToV2:1Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_2:1s
+read_batch_features/read/ReaderReadUpToV2:0Dread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch_1:1p
*read_batch_features/random_shuffle_queue:0Bread_batch_features/cond/random_shuffle_queue_EnqueueMany/Switch:1
�
$read_batch_features/cond/cond_text_1"read_batch_features/cond/pred_id:0#read_batch_features/cond/switch_f:0*z
/read_batch_features/cond/control_dependency_1:0
"read_batch_features/cond/pred_id:0
#read_batch_features/cond/switch_f:0
�
$read_batch_features/cond_1/cond_text$read_batch_features/cond_1/pred_id:0%read_batch_features/cond_1/switch_t:0 *�
/read_batch_features/cond_1/control_dependency:0
$read_batch_features/cond_1/pred_id:0
Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch:1
Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1:1
Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2:1
%read_batch_features/cond_1/switch_t:0
*read_batch_features/random_shuffle_queue:0
-read_batch_features/read/ReaderReadUpToV2_1:0
-read_batch_features/read/ReaderReadUpToV2_1:1w
-read_batch_features/read/ReaderReadUpToV2_1:1Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_2:1w
-read_batch_features/read/ReaderReadUpToV2_1:0Fread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch_1:1r
*read_batch_features/random_shuffle_queue:0Dread_batch_features/cond_1/random_shuffle_queue_EnqueueMany/Switch:1
�
&read_batch_features/cond_1/cond_text_1$read_batch_features/cond_1/pred_id:0%read_batch_features/cond_1/switch_f:0*�
1read_batch_features/cond_1/control_dependency_1:0
$read_batch_features/cond_1/pred_id:0
%read_batch_features/cond_1/switch_f:0
�
$read_batch_features/cond_2/cond_text$read_batch_features/cond_2/pred_id:0%read_batch_features/cond_2/switch_t:0 *�
/read_batch_features/cond_2/control_dependency:0
$read_batch_features/cond_2/pred_id:0
Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch:1
Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1:1
Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2:1
%read_batch_features/cond_2/switch_t:0
*read_batch_features/random_shuffle_queue:0
-read_batch_features/read/ReaderReadUpToV2_2:0
-read_batch_features/read/ReaderReadUpToV2_2:1w
-read_batch_features/read/ReaderReadUpToV2_2:0Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_1:1w
-read_batch_features/read/ReaderReadUpToV2_2:1Fread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch_2:1r
*read_batch_features/random_shuffle_queue:0Dread_batch_features/cond_2/random_shuffle_queue_EnqueueMany/Switch:1
�
&read_batch_features/cond_2/cond_text_1$read_batch_features/cond_2/pred_id:0%read_batch_features/cond_2/switch_f:0*�
1read_batch_features/cond_2/control_dependency_1:0
$read_batch_features/cond_2/pred_id:0
%read_batch_features/cond_2/switch_f:0
�
$read_batch_features/cond_3/cond_text$read_batch_features/cond_3/pred_id:0%read_batch_features/cond_3/switch_t:0 *�
/read_batch_features/cond_3/control_dependency:0
$read_batch_features/cond_3/pred_id:0
Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch:1
Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1:1
Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2:1
%read_batch_features/cond_3/switch_t:0
*read_batch_features/random_shuffle_queue:0
-read_batch_features/read/ReaderReadUpToV2_3:0
-read_batch_features/read/ReaderReadUpToV2_3:1w
-read_batch_features/read/ReaderReadUpToV2_3:0Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_1:1w
-read_batch_features/read/ReaderReadUpToV2_3:1Fread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch_2:1r
*read_batch_features/random_shuffle_queue:0Dread_batch_features/cond_3/random_shuffle_queue_EnqueueMany/Switch:1
�
&read_batch_features/cond_3/cond_text_1$read_batch_features/cond_3/pred_id:0%read_batch_features/cond_3/switch_f:0*�
1read_batch_features/cond_3/control_dependency_1:0
$read_batch_features/cond_3/pred_id:0
%read_batch_features/cond_3/switch_f:0"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0" 
global_step

global_step:0"�
trainable_variables��
�
9linear/text_ids_weighted_by_text_weights/weights/part_0:0>linear/text_ids_weighted_by_text_weights/weights/part_0/Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/read:0"@
0linear/text_ids_weighted_by_text_weights/weights�8  "�8
�
linear/bias_weight/part_0:0 linear/bias_weight/part_0/Assign linear/bias_weight/part_0/read:0"
linear/bias_weight ""!
local_init_op

group_deps_1"�
	variables��
7
global_step:0global_step/Assignglobal_step/read:0
�
9linear/text_ids_weighted_by_text_weights/weights/part_0:0>linear/text_ids_weighted_by_text_weights/weights/part_0/Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/read:0"@
0linear/text_ids_weighted_by_text_weights/weights�8  "�8
�
linear/bias_weight/part_0:0 linear/bias_weight/part_0/Assign linear/bias_weight/part_0/read:0"
linear/bias_weight "
R
train_op/beta1_power:0train_op/beta1_power/Assigntrain_op/beta1_power/read:0
R
train_op/beta2_power:0train_op/beta2_power/Assigntrain_op/beta2_power/read:0
�
>linear/text_ids_weighted_by_text_weights/weights/part_0/Adam:0Clinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/AssignClinear/text_ids_weighted_by_text_weights/weights/part_0/Adam/read:0"E
5linear/text_ids_weighted_by_text_weights/weights/Adam�8  "�8
�
@linear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1:0Elinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/AssignElinear/text_ids_weighted_by_text_weights/weights/part_0/Adam_1/read:0"G
7linear/text_ids_weighted_by_text_weights/weights/Adam_1�8  "�8
�
 linear/bias_weight/part_0/Adam:0%linear/bias_weight/part_0/Adam/Assign%linear/bias_weight/part_0/Adam/read:0""
linear/bias_weight/Adam "
�
"linear/bias_weight/part_0/Adam_1:0'linear/bias_weight/part_0/Adam_1/Assign'linear/bias_weight/part_0/Adam_1/read:0"$
linear/bias_weight/Adam_1 ""
losses

training_loss:0"�
	summaries�
�
9read_batch_features/file_name_queue/fraction_of_32_full:0
1read_batch_features/fraction_over_10_of_10_full:0
_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full:0
training_loss/ScalarSummary:0"
train_op

train_op/Adam"�
local_variables�
�
metrics/accuracy/total:0
metrics/accuracy/count:0
metrics/auc/true_positives:0
metrics/auc/false_negatives:0
metrics/auc/true_negatives:0
metrics/auc/false_positives:0
metrics/mean/total:0
metrics/mean/count:0"d
linearZ
X
9linear/text_ids_weighted_by_text_weights/weights/part_0:0
linear/bias_weight/part_0:0"�
queue_runners��
�
#read_batch_features/file_name_queue?read_batch_features/file_name_queue/file_name_queue_EnqueueMany9read_batch_features/file_name_queue/file_name_queue_Close";read_batch_features/file_name_queue/file_name_queue_Close_1*
�
(read_batch_features/random_shuffle_queue read_batch_features/cond/Merge:0"read_batch_features/cond_1/Merge:0"read_batch_features/cond_2/Merge:0"read_batch_features/cond_3/Merge:0.read_batch_features/random_shuffle_queue_Close"0read_batch_features/random_shuffle_queue_Close_1*
�
read_batch_features/fifo_queue&read_batch_features/fifo_queue_enqueue(read_batch_features/fifo_queue_enqueue_1$read_batch_features/fifo_queue_Close"&read_batch_features/fifo_queue_Close_1*"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"&

summary_op

Merge/MergeSummary:0"
ready_op


concat:0"m
model_variablesZ
X
9linear/text_ids_weighted_by_text_weights/weights/part_0:0
linear/bias_weight/part_0:0"
init_op


group_deps�^�}&       sO� 	#���@�A:TOUT/model/model.ckpt)"��       mS+		j���@�A:/����       �+��	����@�A*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�?@c^3�%       �6�	�����@�Ae*

global_step/sec�b�CD{��       �+��	½��@�Ae*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�@8���&       sO� 	�W���@�A�*

global_step/sec$�9D��Q��       �N�	>^���@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�@��k&       sO� 	�����@�A�*

global_step/sec35?D�����       �N�	����@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��?��A�&       sO� 	�����@�A�*

global_step/secN�FDo���       �N�	2����@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

losseƒ?/I��&       sO� 	��E��@�A�*

global_step/sec^�fB��7��       �N�	A�E��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�W�?f�"�&       sO� 	��P��@�A�*

global_step/secA�D#E7��       �N�	t�P��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss"�?��$�&       sO� 	SZ��@�A�*

global_step/sec��-D�M���       �N�	J
Z��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossʧ`?��&       sO� 	�b��@�A�*

global_step/sech�9D���}�       �N�	�b��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss;��?��$&       sO� 	k�j��@�A�*

global_step/secC�BD9��	�       �N�	��j��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�9�?�6�/&       sO� 	��s��@�A�*

global_step/secdt1DҶJ!�       �N�	a�s��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��d?WU{Q&       sO� 	{1}��@�A�*

global_step/sec�D+DEF�       �N�	�6}��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss���?����&       sO� 	ds���@�A�	*

global_step/sec$�,DԲ���       �N�	:z���@�A�	*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�$�?��&       sO� 	�����@�A�
*

global_step/sec	FD�j��       �N�	\����@�A�
*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossk�?N��&       sO� 	BҖ��@�A�
*

global_step/sec��@Duya��       �N�	�ؖ��@�A�
*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossN�{?�,�&       sO� 	8M���@�A�*

global_step/secC�<D�e���       �N�	.T���@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��j?Wr��&       sO� 	rk���@�A�*

global_step/sec�EDڷ�S�       �N�	r���@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss5��>ըA&       sO� 	e����@�A�*

global_step/sec�BD�+���       �N�	端��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�c?Ӱ�&       sO� 	R����@�A�*

global_step/sec�HD�/�#�       �N�	�����@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�F�>�%�&       sO� 	�ۿ��@�A�*

global_step/sec��AD݂�       �N�	a���@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss
�l?N�D]&       sO� 	�����@�A�*

global_step/sec��3DШV��       �N�	q����@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�<�>�|t&       sO� 	� ���@�A�*

global_step/sec�BD�����       �N�	S���@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�r?�S�*&       sO� 	���@�A�*

global_step/sec�ED��)�       �N�	����@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��?||�a&       sO� 	kJ���@�A�*

global_step/sec�3CD����       �N�	HQ���@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�R�>�B&       sO� 	�|���@�A�*

global_step/sec�3CD��\��       �N�	b����@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

losstȷ>/j�&       sO� 	_����@�A�*

global_step/sec��/D� �	�       �N�	�����@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�n�>�$�&       sO� 	�a���@�A�*

global_step/sec��5D.g�       �N�	�h���@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��=ď�2&       sO� 	� ��@�A�*

global_step/secɓND�V�       �N�	�(��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�qQ>x#~&       sO� 	,g��@�A�*

global_step/secπ,D+����       �N�	�n��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��> W�h&       sO� 	:��@�A�*

global_step/sec��9D>���       �N�	���@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossxv�>�(9&       sO� 	�H��@�A�*

global_step/sec�wAD��� �       �N�	�O��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossE�<���&       sO� 	à%��@�A�*

global_step/sec��?D���       �N�	 �%��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�r!>V%a�&       sO� 	��-��@�A�*

global_step/sec�x?DJ�)��       �N�	.��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss���>mD��&       sO� 	�}6��@�A�*

global_step/sec:<D��@�       �N�	��6��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss���>+��&       sO� 	T�?��@�A�*

global_step/seca*Dc-�2�       �N�	��?��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossڌ�>�|��&       sO� 	�H��@�A�*

global_step/sec	[EDq��<�       �N�	�H��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossb�;>��^�&       sO� 	��P��@�A�*

global_step/secY�:D�^���       �N�	g�P��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss>=�Zl�&       sO� 	@�X��@�A�*

global_step/sec�TAD�*f��       �N�	L�X��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�0�>�yo�&       sO� 	L4a��@�A�*

global_step/sec$�?D#z-��       �N�	�;a��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossX%F>k��&       sO� 	�"j��@�A�*

global_step/sec%3D�#j�       �N�	�'j��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��X<[���&       sO� 	#�r��@�A�*

global_step/sec��7DD�c��       �N�	C�r��@�A�*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�>�&��&       sO� 	��{��@�A� *

global_step/sec��3D�h	?�       �N�	7�{��@�A� *�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�Pk=��&       sO� 	����@�A� *

global_step/sec��,D-&��       �N�	F���@�A� *�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�i>�|$�&       sO� 	�����@�A�!*

global_step/secZ�HD�ZRf�       �N�	6���@�A�!*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�7�>��Ņ&       sO� 	T;���@�A�"*

global_step/sec��AD[��       �N�	�A���@�A�"*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossP0>���'&       sO� 	Ul���@�A�#*

global_step/secRRCD
�vH�       �N�	�r���@�A�#*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss(�T>���,&       sO� 	7l���@�A�#*

global_step/sec�HD��p�       �N�	qq���@�A�#*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�Gb>c��&       sO� 	­��@�A�$*

global_step/sec��?DA�m�       �N�	�ƭ��@�A�$*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�`A>���F&       sO� 	^ڶ��@�A�%*

global_step/secp�/Dy3.��       �N�	����@�A�%*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossUB>ž��&       sO� 	�_���@�A�&*

global_step/sec�;D	���       �N�	
g���@�A�&*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��y>B}�'&       sO� 	�����@�A�'*

global_step/sec)>D�}��       �N�	�����@�A�'*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossG>��W�&       sO� 	����@�A�'*

global_step/sec�1GD�IQ�       �N�	�����@�A�'*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�\r<Ԫ�&       sO� 	v����@�A�(*

global_step/sec��CD�.�@�       �N�	����@�A�(*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�)B>Hk�&       sO� 	�����@�A�)*

global_step/sec��7D�zr��       �N�	����@�A�)*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�B>�y��&       sO� 	0����@�A�**

global_step/sec�DDe���       �N�	����@�A�**�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��=���Q&       sO� 	�����@�A�**

global_step/sectI8D.�C�       �N�	Y����@�A�**�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�/>����&       sO� 	V����@�A�+*

global_step/sec�)D.�#�       �N�	�����@�A�+*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�-�>�Z�&       sO� 	C���@�A�,*

global_step/sec��$DCv���       �N�	���@�A�,*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossR��=q�G�&       sO� 	&���@�A�-*

global_step/sec s4D%`�m�       �N�	���@�A�-*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossD��=�t��&       sO� 	���@�A�.*

global_step/sec��HD�yL�       �N�	r���@�A�.*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossDX?���&       sO� 	{���@�A�.*

global_step/sec��GD��;��       �N�	����@�A�.*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�>>����&       sO� 	+&��@�A�/*

global_step/secw�:D/3���       �N�	&��@�A�/*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�%1>)�3&       sO� 	�d.��@�A�0*

global_step/sec��@D� �       �N�	�k.��@�A�0*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossj<>� P�&       sO� 	�\6��@�A�1*

global_step/sec��HDW�E��       �N�	�a6��@�A�1*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

losss�i=@��j&       sO� 	@?��@�A�2*

global_step/sec97DVκ	�       �N�	??��@�A�2*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss���=1�&       sO� 	� H��@�A�2*

global_step/sec�3D����       �N�	hH��@�A�2*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossa8=v�1-&       sO� 	��Q��@�A�3*

global_step/secPP(D�N
��       �N�	n�Q��@�A�3*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossf�=r��l&       sO� 	]Z��@�A�4*

global_step/secEn:D6�y�       �N�	 Z��@�A�4*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossj�=#�d&       sO� 	&6b��@�A�5*

global_step/sec>
ED�����       �N�	�<b��@�A�5*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�l�<���&       sO� 	�}j��@�A�5*

global_step/sec�9AD<����       �N�	��j��@�A�5*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossx�N=S�g&       sO� 	_�s��@�A�6*

global_step/sec��+D)rhB�       �N�	��s��@�A�6*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��}=gN�&       sO� 	�|��@�A�7*

global_step/sec�S6D��q�       �N�	b�|��@�A�7*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��=�;�&       sO� 	
M���@�A�8*

global_step/sec*�$DK����       �N�	~T���@�A�8*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�#=����&       sO� 	~����@�A�9*

global_step/secPs?D�Ø�       �N�	X����@�A�9*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��Z<���&       sO� 	1���@�A�9*

global_step/sec;DjDC�       �N�	�7���@�A�9*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss >��|�&       sO� 	�	���@�A�:*

global_step/sec��"D;U�a�       �N�	����@�A�:*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossbab<q���&       sO� 	�S���@�A�;*

global_step/sec� ADX=�1�       �N�	�Z���@�A�;*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�7!=,)�&       sO� 	�Ա��@�A�<*

global_step/sec� <D�H�       �N�	ܱ��@�A�<*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�(<���&       sO� 	�2���@�A�<*

global_step/secOC?D`����       �N�	i9���@�A�<*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�v=��_'&       sO� 	J����@�A�=*

global_step/sec*x*DΰŰ�       �N�	����@�A�=*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossa|�=��9&       sO� 	� ���@�A�>*

global_step/secN>;Dְ�$�       �N�	�'���@�A�>*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�o�=n{y�&       sO� 	i���@�A�?*

global_step/secH%AD�5>�       �N�	�o���@�A�?*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��<�0�X&       sO� 	h����@�A�@*

global_step/sec�(<Da���       �N�	c����@�A�@*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�"p=�P�g&       sO� 	�����@�A�@*

global_step/sect�7D0-ʡ�       �N�	����@�A�@*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss6�;u��&       sO� 	aU���@�A�A*

global_step/sec��$D{����       �N�	�[���@�A�A*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossc�<��U�&       sO� 	4����@�A�B*

global_step/sec?<D�X��       �N�	�����@�A�B*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossUK�<6��&       sO� 	c ��@�A�C*

global_step/sec#;D�;�,�       �N�	�i ��@�A�C*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��>bw�&       sO� 	�	��@�A�C*

global_step/sec�9Dy"-��       �N�	�	��@�A�C*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss��<V�Q�&       sO� 	�R��@�A�D*

global_step/sec�.,D����       �N�	�Y��@�A�D*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossV�_<���0&       sO� 	^,��@�A�E*

global_step/sec��4D@cE�       �N�	73��@�A�E*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�ބ<�s&       sO� 	%�#��@�A�F*

global_step/sec�Z9D
�}*�       �N�	��#��@�A�F*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�f�<X���&       sO� 	7Q,��@�A�G*

global_step/sec`�;D�*���       �N�	�W,��@�A�G*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss2<D>Jqx�&       sO� 	4�5��@�A�G*

global_step/sec�.D�{P]�       �N�	��5��@�A�G*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full�p}?

loss/<�f�O&       sO� 	pC>��@�A�H*

global_step/sec��6D�Ќ�       �N�	�J>��@�A�H*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�[=vTG&       sO� 	�F��@�A�I*

global_step/sec��=D��?��       �N�	
�F��@�A�I*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss;�	>�dM�&       sO� 	?�O��@�A�J*

global_step/sec��-D�$�<�       �N�	��O��@�A�J*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�O=2]��&       sO� 	��X��@�A�K*

global_step/sec�0D5�       �N�	��X��@�A�K*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�T�<1�	2&       sO� 	�Tb��@�A�K*

global_step/sec�`*D�XM�       �N�	�[b��@�A�K*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss�Ld=��SN&       sO� 	��j��@�A�L*

global_step/sec(�>D��5��       �N�	�j��@�A�L*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

loss5�I=���(&       sO� 	��r��@�A�M*

global_step/sec�wAD��>�       �N�	�s��@�A�M*�
>
7read_batch_features/file_name_queue/fraction_of_32_full  �?
6
/read_batch_features/fraction_over_10_of_10_full  �?
d
]read_batch_features/queue/parsed_features/read_batch_features/fifo_queue/fraction_of_100_full  �?

lossj�=0n.'       ��F	�����@�A�N:TOUT/model/model.ckptQ���