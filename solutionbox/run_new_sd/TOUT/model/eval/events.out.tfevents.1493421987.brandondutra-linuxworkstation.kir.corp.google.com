       £K"	  јиу@÷Abrain.Event:2Љ'–Пџ     z Ч	Сниу@÷A"ВЈ

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
П
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
≤
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
У
)read_batch_features/file_name_queue/inputConst*
dtype0*µ
valueЂB®B.exout/features_test-00000-of-00006.tfrecord.gzB.exout/features_test-00001-of-00006.tfrecord.gzB.exout/features_test-00002-of-00006.tfrecord.gzB.exout/features_test-00003-of-00006.tfrecord.gzB.exout/features_test-00004-of-00006.tfrecord.gzB.exout/features_test-00005-of-00006.tfrecord.gz*
_output_shapes
:
j
(read_batch_features/file_name_queue/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
o
-read_batch_features/file_name_queue/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
∞
+read_batch_features/file_name_queue/GreaterGreater(read_batch_features/file_name_queue/Size-read_batch_features/file_name_queue/Greater/y*
T0*
_output_shapes
: 
І
0read_batch_features/file_name_queue/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
ѓ
8read_batch_features/file_name_queue/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
њ
1read_batch_features/file_name_queue/Assert/AssertAssert+read_batch_features/file_name_queue/Greater8read_batch_features/file_name_queue/Assert/Assert/data_0*
	summarize*

T
2
Љ
,read_batch_features/file_name_queue/IdentityIdentity)read_batch_features/file_name_queue/input2^read_batch_features/file_name_queue/Assert/Assert*
T0*
_output_shapes
:
x
6read_batch_features/file_name_queue/limit_epochs/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ы
7read_batch_features/file_name_queue/limit_epochs/epochs
VariableV2*
dtype0	*
shape: *
	container *
shared_name *
_output_shapes
: 
ѕ
>read_batch_features/file_name_queue/limit_epochs/epochs/AssignAssign7read_batch_features/file_name_queue/limit_epochs/epochs6read_batch_features/file_name_queue/limit_epochs/Const*
validate_shape(*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
use_locking(*
T0	*
_output_shapes
: 
о
<read_batch_features/file_name_queue/limit_epochs/epochs/readIdentity7read_batch_features/file_name_queue/limit_epochs/epochs*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
T0	*
_output_shapes
: 
ъ
:read_batch_features/file_name_queue/limit_epochs/CountUpTo	CountUpTo7read_batch_features/file_name_queue/limit_epochs/epochs*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
limit*
T0	*
_output_shapes
: 
ћ
0read_batch_features/file_name_queue/limit_epochsIdentity,read_batch_features/file_name_queue/Identity;^read_batch_features/file_name_queue/limit_epochs/CountUpTo*
T0*
_output_shapes
:
®
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
Ё
?read_batch_features/file_name_queue/file_name_queue_EnqueueManyQueueEnqueueManyV2#read_batch_features/file_name_queue0read_batch_features/file_name_queue/limit_epochs*

timeout_ms€€€€€€€€€*
Tcomponents
2
Н
9read_batch_features/file_name_queue/file_name_queue_CloseQueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues( 
П
;read_batch_features/file_name_queue/file_name_queue_Close_1QueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues(
Д
8read_batch_features/file_name_queue/file_name_queue_SizeQueueSizeV2#read_batch_features/file_name_queue*
_output_shapes
: 
Ъ
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
§
'read_batch_features/file_name_queue/mulMul(read_batch_features/file_name_queue/Cast)read_batch_features/file_name_queue/mul/y*
T0*
_output_shapes
: 
і
<read_batch_features/file_name_queue/fraction_of_32_full/tagsConst*
dtype0*H
value?B= B7read_batch_features/file_name_queue/fraction_of_32_full*
_output_shapes
: 
–
7read_batch_features/file_name_queue/fraction_of_32_fullScalarSummary<read_batch_features/file_name_queue/fraction_of_32_full/tags'read_batch_features/file_name_queue/mul*
T0*
_output_shapes
: 
Х
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
ш
)read_batch_features/read/ReaderReadUpToV2ReaderReadUpToV2)read_batch_features/read/TFRecordReaderV2#read_batch_features/file_name_queue5read_batch_features/read/ReaderReadUpToV2/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
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
ю
+read_batch_features/read/ReaderReadUpToV2_1ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_1#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_1/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
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
ю
+read_batch_features/read/ReaderReadUpToV2_2ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_2#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_2/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
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
ю
+read_batch_features/read/ReaderReadUpToV2_3ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_3#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_3/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
[
read_batch_features/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
¶
read_batch_features/fifo_queueFIFOQueueV2*
capacity*
_output_shapes
: *
shapes
: : *
component_types
2*
	container *
shared_name 
В
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
Ў
6read_batch_features/cond/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue read_batch_features/cond/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
T0*
_output_shapes
: : 
К
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1Switch)read_batch_features/read/ReaderReadUpToV2 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
М
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2Switch+read_batch_features/read/ReaderReadUpToV2:1 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
©
/read_batch_features/cond/fifo_queue_EnqueueManyQueueEnqueueManyV28read_batch_features/cond/fifo_queue_EnqueueMany/Switch:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms€€€€€€€€€*
Tcomponents
2
г
+read_batch_features/cond/control_dependencyIdentity!read_batch_features/cond/switch_t0^read_batch_features/cond/fifo_queue_EnqueueMany*4
_class*
(&loc:@read_batch_features/cond/switch_t*
T0
*
_output_shapes
: 
I
read_batch_features/cond/NoOpNoOp"^read_batch_features/cond/switch_f
”
-read_batch_features/cond/control_dependency_1Identity!read_batch_features/cond/switch_f^read_batch_features/cond/NoOp*4
_class*
(&loc:@read_batch_features/cond/switch_f*
T0
*
_output_shapes
: 
ѓ
read_batch_features/cond/MergeMerge-read_batch_features/cond/control_dependency_1+read_batch_features/cond/control_dependency*
_output_shapes
: : *
T0
*
N
Д
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
№
8read_batch_features/cond_1/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_1/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
T0*
_output_shapes
: : 
Т
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_1:1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_1/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms€€€€€€€€€*
Tcomponents
2
л
-read_batch_features/cond_1/control_dependencyIdentity#read_batch_features/cond_1/switch_t2^read_batch_features/cond_1/fifo_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_1/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_1/NoOpNoOp$^read_batch_features/cond_1/switch_f
џ
/read_batch_features/cond_1/control_dependency_1Identity#read_batch_features/cond_1/switch_f ^read_batch_features/cond_1/NoOp*6
_class,
*(loc:@read_batch_features/cond_1/switch_f*
T0
*
_output_shapes
: 
µ
 read_batch_features/cond_1/MergeMerge/read_batch_features/cond_1/control_dependency_1-read_batch_features/cond_1/control_dependency*
_output_shapes
: : *
T0
*
N
Д
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
№
8read_batch_features/cond_2/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_2/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
T0*
_output_shapes
: : 
Т
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_2"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_2:1"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_2/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms€€€€€€€€€*
Tcomponents
2
л
-read_batch_features/cond_2/control_dependencyIdentity#read_batch_features/cond_2/switch_t2^read_batch_features/cond_2/fifo_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_2/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_2/NoOpNoOp$^read_batch_features/cond_2/switch_f
џ
/read_batch_features/cond_2/control_dependency_1Identity#read_batch_features/cond_2/switch_f ^read_batch_features/cond_2/NoOp*6
_class,
*(loc:@read_batch_features/cond_2/switch_f*
T0
*
_output_shapes
: 
µ
 read_batch_features/cond_2/MergeMerge/read_batch_features/cond_2/control_dependency_1-read_batch_features/cond_2/control_dependency*
_output_shapes
: : *
T0
*
N
Д
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
№
8read_batch_features/cond_3/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_3/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
T0*
_output_shapes
: : 
Т
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_3"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_3:1"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_3/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms€€€€€€€€€*
Tcomponents
2
л
-read_batch_features/cond_3/control_dependencyIdentity#read_batch_features/cond_3/switch_t2^read_batch_features/cond_3/fifo_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_3/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_3/NoOpNoOp$^read_batch_features/cond_3/switch_f
џ
/read_batch_features/cond_3/control_dependency_1Identity#read_batch_features/cond_3/switch_f ^read_batch_features/cond_3/NoOp*6
_class,
*(loc:@read_batch_features/cond_3/switch_f*
T0
*
_output_shapes
: 
µ
 read_batch_features/cond_3/MergeMerge/read_batch_features/cond_3/control_dependency_1-read_batch_features/cond_3/control_dependency*
_output_shapes
: : *
T0
*
N
s
$read_batch_features/fifo_queue_CloseQueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues( 
u
&read_batch_features/fifo_queue_Close_1QueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues(
j
#read_batch_features/fifo_queue_SizeQueueSizeV2read_batch_features/fifo_queue*
_output_shapes
: 
u
read_batch_features/CastCast#read_batch_features/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
^
read_batch_features/mul/yConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
t
read_batch_features/mulMulread_batch_features/Castread_batch_features/mul/y*
T0*
_output_shapes
: 
Ф
,read_batch_features/fraction_of_20_full/tagsConst*
dtype0*8
value/B- B'read_batch_features/fraction_of_20_full*
_output_shapes
: 
†
'read_batch_features/fraction_of_20_fullScalarSummary,read_batch_features/fraction_of_20_full/tagsread_batch_features/mul*
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
 
read_batch_featuresQueueDequeueUpToV2read_batch_features/fifo_queueread_batch_features/n*

timeout_ms€€€€€€€€€*
component_types
2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
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
Љ
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
Д
;read_batch_features/ParseExample/ParseExample/sparse_keys_0Const*
dtype0*
valueB Btext_ids*
_output_shapes
: 
И
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
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_1Const*
dtype0*
valueB Btarget*
_output_shapes
: 
±
-read_batch_features/ParseExample/ParseExampleParseExampleread_batch_features:13read_batch_features/ParseExample/ParseExample/names;read_batch_features/ParseExample/ParseExample/sparse_keys_0;read_batch_features/ParseExample/ParseExample/sparse_keys_1:read_batch_features/ParseExample/ParseExample/dense_keys_0:read_batch_features/ParseExample/ParseExample/dense_keys_1(read_batch_features/ParseExample/Reshape&read_batch_features/ParseExample/Const*
dense_shapes
: : *В
_output_shapesp
n:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::€€€€€€€€€:€€€€€€€€€*
Ndense*
sparse_types
2	*
Tdense
2		*
Nsparse
Ђ
 read_batch_features/fifo_queue_1FIFOQueueV2*
capacityd*
_output_shapes
: *
shapes
 * 
component_types
2								*
	container *
shared_name 
n
%read_batch_features/fifo_queue_1_SizeQueueSizeV2 read_batch_features/fifo_queue_1*
_output_shapes
: 
y
read_batch_features/Cast_1Cast%read_batch_features/fifo_queue_1_Size*

DstT0*

SrcT0*
_output_shapes
: 
`
read_batch_features/mul_1/yConst*
dtype0*
valueB
 *
„#<*
_output_shapes
: 
z
read_batch_features/mul_1Mulread_batch_features/Cast_1read_batch_features/mul_1/y*
T0*
_output_shapes
: 
Д
dread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full/tagsConst*
dtype0*p
valuegBe B_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full*
_output_shapes
: 
Т
_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_fullScalarSummarydread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full/tagsread_batch_features/mul_1*
T0*
_output_shapes
: 
∞
(read_batch_features/fifo_queue_1_enqueueQueueEnqueueV2 read_batch_features/fifo_queue_1/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms€€€€€€€€€*
Tcomponents
2								
≤
*read_batch_features/fifo_queue_1_enqueue_1QueueEnqueueV2 read_batch_features/fifo_queue_1/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms€€€€€€€€€*
Tcomponents
2								
w
&read_batch_features/fifo_queue_1_CloseQueueCloseV2 read_batch_features/fifo_queue_1*
cancel_pending_enqueues( 
y
(read_batch_features/fifo_queue_1_Close_1QueueCloseV2 read_batch_features/fifo_queue_1*
cancel_pending_enqueues(
≠
(read_batch_features/fifo_queue_1_DequeueQueueDequeueV2 read_batch_features/fifo_queue_1*

timeout_ms€€€€€€€€€* 
component_types
2								*С
_output_shapes
}:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€::€€€€€€€€€:€€€€€€€€€::€€€€€€€€€
Y
ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Р

ExpandDims
ExpandDims(read_batch_features/fifo_queue_1_DequeueExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:€€€€€€€€€
[
ExpandDims_1/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Ц
ExpandDims_1
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:1ExpandDims_1/dim*

Tdim0*
T0	*'
_output_shapes
:€€€€€€€€€
V
linear/linear/mod/yConst*
dtype0	*
value
B	 RЛ8*
_output_shapes
: 
М
linear/linear/modFloorMod*read_batch_features/fifo_queue_1_Dequeue:3linear/linear/mod/y*
T0	*#
_output_shapes
:€€€€€€€€€
м
Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/ConstConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB	Л8*    *
_output_shapes
:	Л8
щ
7linear/text_ids_weighted_by_text_weights/weights/part_0
VariableV2*
	container *
_output_shapes
:	Л8*
dtype0*
shape:	Л8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
л
>linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign7linear/text_ids_weighted_by_text_weights/weights/part_0Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/Const*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	Л8
ч
<linear/text_ids_weighted_by_text_weights/weights/part_0/readIdentity7linear/text_ids_weighted_by_text_weights/weights/part_0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	Л8
ѓ
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
Ѓ
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
У
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SliceSlice*read_batch_features/fifo_queue_1_Dequeue:4elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/begindlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
©
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
ж
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ProdProd_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
™
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
ѕ
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GatherGather*read_batch_features/fifo_queue_1_Dequeue:4hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
х
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shapePack^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Prod`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather*
N*
T0	*
_output_shapes
:*

axis 
т
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeSparseReshape*read_batch_features/fifo_queue_1_Dequeue:2*read_batch_features/fifo_queue_1_Dequeue:4qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shape*-
_output_shapes
:€€€€€€€€€:
љ
plinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/IdentityIdentitylinear/linear/mod*
T0	*#
_output_shapes
:€€€€€€€€€
™
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
А
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualGreaterEqualplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityhlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
®
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Ђ
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterGreater*read_batch_features/fifo_queue_1_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/y*
T0*#
_output_shapes
:€€€€€€€€€
в
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd
LogicalAndflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater*#
_output_shapes
:€€€€€€€€€
ч
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/WhereWheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:€€€€€€€€€
Ї
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
т
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ReshapeReshape_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Whereglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	*
Tshape0
Ш
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:€€€€€€€€€
Э
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2Gatherplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:€€€€€€€€€
ю
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityIdentityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
щ
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1Wheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:€€€€€€€€€
Љ
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
ш
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1Reshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shape*#
_output_shapes
:€€€€€€€€€*
T0	*
Tshape0
Ъ
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:€€€€€€€€€
ў
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4Gather*read_batch_features/fifo_queue_1_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
А
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Identityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
µ
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
ћ
Бlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
ќ
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
ќ
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
С
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_sliceStridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityБlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stackГlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
І
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/CastCast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
ї
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
ї
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/rangeRangeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startrlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Castylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
Ѓ
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1Castslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:€€€€€€€€€
’
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
„
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
„
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
¶
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stackЕlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
√
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffListDifftlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Щ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityГlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stackЕlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
«
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
і
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims
ExpandDims}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
ћ
Йlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
ћ
Йlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
м
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseSparseToDensevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDimsЙlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_valuesЙlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:€€€€€€€€€
ћ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
µ
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ReshapeReshapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0	*
Tshape0
Ѓ
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_like	ZerosLikeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:€€€€€€€€€
ї
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
≥
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concatConcatV2ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_likeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axis*
N*

Tidx0*'
_output_shapes
:€€€€€€€€€*
T0	
©
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ShapeShapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
Т
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/FillFillslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Shapeslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:€€€€€€€€€
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
†
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*'
_output_shapes
:€€€€€€€€€*
T0	
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ъ
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Fill{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axis*
N*

Tidx0*#
_output_shapes
:€€€€€€€€€*
T0	
°
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorderSparseReordervlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
Л
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/IdentityIdentityblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*
_output_shapes
:
Ї
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
ќ
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ы
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_sliceStridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stackЕlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
Ђ
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/CastCast}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Я
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/rangeRange{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/starttlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
≤
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1Castulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range*

DstT0	*

SrcT0*#
_output_shapes
:€€€€€€€€€
„
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
ў
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
ў
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ѓ
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stackЗlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
…
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffListDiffvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
“
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
“
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
£
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2StridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stackЗlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
…
~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Ї
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims
ExpandDimslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
ќ
Лlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
ќ
Лlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
ц
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDenseSparseToDensexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffzlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDimsЛlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesЛlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:€€€€€€€€€
ќ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
ї
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ReshapeReshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0	*
Tshape0
≤
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like	ZerosLikewlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape*
T0	*'
_output_shapes
:€€€€€€€€€
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
ї
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshapezlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axis*
N*

Tidx0*'
_output_shapes
:€€€€€€€€€*
T0	
≠
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ShapeShapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff*
out_type0*
T0	*
_output_shapes
:
Ш
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/FillFillulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Shapeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Const*
T0*#
_output_shapes
:€€€€€€€€€
њ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
¶
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axis*
N*

Tidx0*'
_output_shapes
:€€€€€€€€€*
T0	
њ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
†
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axis*
N*

Tidx0*#
_output_shapes
:€€€€€€€€€*
T0
©
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorderSparseReorderxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
П
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/IdentityIdentitydlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0	*
_output_shapes
:
„
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
ў
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
ў
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
«
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_sliceStridedSlice{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorderЕlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stackЗlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
Љ
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/CastCastlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€
ц
Вlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupGather<linear/text_ids_weighted_by_text_weights/weights/part_0/read}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
validate_indices(*
Tparams0*
Tindices0	*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*'
_output_shapes
:€€€€€€€€€
Є
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/RankConst*
dtype0*
value	B :*
_output_shapes
: 
є
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
О
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/subSubvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Rankwlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/y*
T0*
_output_shapes
: 
√
Аlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
µ
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims
ExpandDimsulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/subАlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Њ
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
®
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/FillFill|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/value*
T0*#
_output_shapes
:€€€€€€€€€
ґ
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ShapeShapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1*
out_type0*
T0*
_output_shapes
:
њ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ј
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Shapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axis*
N*

Tidx0*#
_output_shapes
:€€€€€€€€€*
T0
њ
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ReshapeReshapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Ѓ
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulMulВlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:€€€€€€€€€
∞
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse
SegmentSumulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:€€€€€€€€€
Ї
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
Ц
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2Reshape{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shape*'
_output_shapes
:€€€€€€€€€*
T0
*
Tshape0
Р
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ShapeShapeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Ј
mlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
є
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
є
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ї
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
£
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
н
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stackPackalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
щ
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/TileTileclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ц
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like	ZerosLikeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
ќ
Ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weightsSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tiledlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_likeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
∆
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/CastCast*read_batch_features/fifo_queue_1_Dequeue:4*

DstT0*

SrcT0	*
_output_shapes
:
±
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
∞
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Ќ
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1Slice^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Castglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ъ
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1ShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
±
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
є
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/sizeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
–
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2Slicealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
І
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
”
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concatConcatV2alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
л
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3ReshapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
l
linear/linear/Reshape/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
в
linear/linear/ReshapeReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3linear/linear/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
¶
+linear/bias_weight/part_0/Initializer/ConstConst*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
valueB*    *
_output_shapes
:
≥
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
о
 linear/bias_weight/part_0/AssignAssignlinear/bias_weight/part_0+linear/bias_weight/part_0/Initializer/Const*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
Ш
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
Ф
linear/linear/BiasAddBiasAddlinear/linear/Reshapelinear/bias_weight*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
m
predictions/probabilitiesSoftmaxlinear/linear/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
_
predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
Н
predictions/classesArgMaxlinear/linear/BiasAddpredictions/classes/dimension*#
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
О
0training_loss/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
.training_loss/softmax_cross_entropy_loss/ShapeShape0training_loss/softmax_cross_entropy_loss/Squeeze*
out_type0*
T0	*
_output_shapes
:
и
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
]
training_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Т
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
У
,metrics/remove_squeezable_dimensions/SqueezeSqueezeExpandDims_1*
squeeze_dims

€€€€€€€€€*
T0	*#
_output_shapes
:€€€€€€€€€
З
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:€€€€€€€€€
c
metrics/ToFloatCastmetrics/Equal*

DstT0*

SrcT0
*#
_output_shapes
:€€€€€€€€€
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
ћ
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
validate_shape(*)
_class
loc:@metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
Л
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
ќ
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
validate_shape(*)
_class
loc:@metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
Л
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
T0*
_output_shapes
: 
_
metrics/accuracy/SizeSizemetrics/ToFloat*
out_type0*
T0*
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
В
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
і
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*)
_class
loc:@metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
Љ
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
П
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
В
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
А
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
Ы
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
Т
metrics/Assert/ConstConst*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
Ъ
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
А
metrics/Reshape/shapeConst^metrics/Assert/Assert*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
{
metrics/ReshapeReshapeExpandDims_1metrics/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	*
Tshape0
]
metrics/one_hot/on_valueConst*
dtype0*
valueB
 *  А?*
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
«
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
axis€€€€€€€€€*
T0*'
_output_shapes
:€€€€€€€€€*
TI0	
f
metrics/CastCastmetrics/one_hot*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€
j
metrics/auc/Reshape/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
Ф
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"   €€€€*
_output_shapes
:
Л
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
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
µ
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
А
metrics/auc/ConstConst*
dtype0*є
valueѓBђ»"†Хњ÷≥ѕ©§;ѕ©$<Јюv<ѕ©§<C‘Ќ<Јюц<Х=ѕ©$=	?9=C‘M=}ib=Јюv=ш…Е=ХР=2_Ъ=ѕ©§=lфЃ=	?є=¶Й√=C‘Ќ=аЎ=}iв=ім=Јюц=™§ >ш…>Gп
>Х>д9>2_>БД>ѕ©$>ѕ)>lф.>ї4>	?9>Wd>>¶ЙC>фЃH>C‘M>СщR>аX>.D]>}ib>ЋОg>іl>hўq>Јюv>$|>™§А>Q7Г>ш…Е>†\И>GпК>оБН>ХР><ІТ>д9Х>ЛћЧ>2_Ъ>ўсЬ>БДЯ>(Ґ>ѕ©§>v<І>ѕ©>≈aђ>lфЃ>З±>їі>bђґ>	?є>∞—ї>WdЊ>€цј>¶Й√>M∆>фЃ»>ЬAЋ>C‘Ќ>кf–>Сщ“>9М’>аЎ>З±Џ>.DЁ>÷÷я>}iв>$ьд>ЋОз>r!к>ім>ЅFп>hўс>lф>Јюц>^Сщ>$ь>ђґю>™§ ?эн?Q7?•А?ш…?L?†\?у•	?Gп
?Ъ8?оБ?BЋ?Х?й]?<І?Рр?д9?7Г?Лћ?я?2_?Ж®?ўс?-;?БД?‘Ќ ?("?{`#?ѕ©$?#у%?v<'? Е(?ѕ)?q+?≈a,?Ђ-?lф.?ј=0?З1?g–2?ї4?c5?bђ6?µх7?	?9?]И:?∞—;?=?Wd>?Ђ≠??€ц@?R@B?¶ЙC?ъ“D?MF?°eG?фЃH?HшI?ЬAK?пКL?C‘M?ЧO?кfP?>∞Q?СщR?еBT?9МU?М’V?аX?3hY?З±Z?џъ[?.D]?ВН^?÷÷_?) a?}ib?–≤c?$ьd?xEf?ЋОg?Ўh?r!j?∆jk?іl?mэm?ЅFo?Рp?hўq?Љ"s?lt?cµu?Јюv?
Hx?^Сy?≤Џz?$|?Ym}?ђґ~? А?*
_output_shapes	
:»
d
metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
Й
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	»
U
metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
Г
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
N*
T0*
_output_shapes
:*

axis 
И
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:»€€€€€€€€€
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
Ѓ
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
T0*
_output_shapes
:
У
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:€€€€€€€€€
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"»      *
_output_shapes
:
Ф
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:»€€€€€€€€€
w
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*(
_output_shapes
:»€€€€€€€€€
c
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"»      *
_output_shapes
:
Ф
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:»€€€€€€€€€
d
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2*(
_output_shapes
:»€€€€€€€€€
`
metrics/auc/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»
И
metrics/auc/true_positives
VariableV2*
dtype0*
shape:»*
	container *
shared_name *
_output_shapes	
:»
Ў
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:»
Ь
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
T0*
_output_shapes	
:»
w
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
w
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

DstT0*

SrcT0
*(
_output_shapes
:»€€€€€€€€€
c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
У
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*
_output_shapes	
:»*
T0*
	keep_dims( *

Tidx0
Ј
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*-
_class#
!loc:@metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:»
b
metrics/auc/zeros_1Const*
dtype0*
valueB»*    *
_output_shapes	
:»
Й
metrics/auc/false_negatives
VariableV2*
dtype0*
shape:»*
	container *
shared_name *
_output_shapes	
:»
Ё
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
validate_shape(*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:»
Я
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
_output_shapes	
:»
|
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

DstT0*

SrcT0
*(
_output_shapes
:»€€€€€€€€€
e
#metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*
_output_shapes	
:»*
T0*
	keep_dims( *

Tidx0
љ
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:»
b
metrics/auc/zeros_2Const*
dtype0*
valueB»*    *
_output_shapes	
:»
И
metrics/auc/true_negatives
VariableV2*
dtype0*
shape:»*
	container *
shared_name *
_output_shapes	
:»
Џ
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:»
Ь
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
_output_shapes	
:»
В
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

DstT0*

SrcT0
*(
_output_shapes
:»€€€€€€€€€
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*
_output_shapes	
:»*
T0*
	keep_dims( *

Tidx0
ї
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:»
b
metrics/auc/zeros_3Const*
dtype0*
valueB»*    *
_output_shapes	
:»
Й
metrics/auc/false_positives
VariableV2*
dtype0*
shape:»*
	container *
shared_name *
_output_shapes	
:»
Ё
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:»
Я
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
T0*
_output_shapes	
:»

metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

DstT0*

SrcT0
*(
_output_shapes
:»€€€€€€€€€
e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*
_output_shapes	
:»*
T0*
	keep_dims( *

Tidx0
љ
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*.
_class$
" loc:@metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:»
V
metrics/auc/add/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:»
Б
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:»
X
metrics/auc/add_2/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
T0*
_output_shapes	
:»
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
T0*
_output_shapes	
:»
Б
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:»
X
metrics/auc/add_4/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:»
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:»
k
!metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:«*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
¬
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

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
¬
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:«
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:«*
_output_shapes
:
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ј
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

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
ј
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:«
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
:«
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:«
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
 *љ7Ж5*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
T0*
_output_shapes	
:»
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:»
X
metrics/auc/add_8/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:»
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
T0*
_output_shapes	
:»
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:»
Y
metrics/auc/add_10/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:»
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
T0*
_output_shapes	
:»
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:«*
_output_shapes
:
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
¬
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

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
¬
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
T0*
_output_shapes	
:«
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:«*
_output_shapes
:
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
¬
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

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
¬
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
T0*
_output_shapes	
:«
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
:«
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:«
]
metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
В
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
И
*metrics/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
Т
(metrics/softmax_cross_entropy_loss/ShapeShape*metrics/softmax_cross_entropy_loss/Squeeze*
out_type0*
T0	*
_output_shapes
:
№
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
a
metrics/eval_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ф
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
Љ
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
Њ
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
§
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
ђ
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
Л
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
`

group_depsNoOp^metrics/mean/update_op^metrics/auc/update_op^metrics/accuracy/update_op
\
eval_step/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
m
	eval_step
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
¶
eval_step/AssignAssign	eval_stepeval_step/initial_value*
validate_shape(*
_class
loc:@eval_step*
use_locking(*
T0*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
T0*
_output_shapes
: 
T
AssignAdd/valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
use_locking( *
T0*
_output_shapes
: 
Е
initNoOp^global_step/Assign?^linear/text_ids_weighted_by_text_weights/weights/part_0/Assign!^linear/bias_weight/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
щ
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized7linear/text_ids_weighted_by_text_weights/weights/part_0*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
љ
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedlinear/bias_weight/part_0*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
щ
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized7read_batch_features/file_name_queue/limit_epochs/epochs*
dtype0	*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
_output_shapes
: 
Ј
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedmetrics/accuracy/total*
dtype0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
Ј
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedmetrics/accuracy/count*
dtype0*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: 
њ
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedmetrics/auc/true_positives*
dtype0*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes
: 
Ѕ
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedmetrics/auc/false_negatives*
dtype0*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes
: 
њ
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializedmetrics/auc/true_negatives*
dtype0*-
_class#
!loc:@metrics/auc/true_negatives*
_output_shapes
: 
Ѕ
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedmetrics/auc/false_positives*
dtype0*.
_class$
" loc:@metrics/auc/false_positives*
_output_shapes
: 
∞
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedmetrics/mean/total*
dtype0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
∞
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedmetrics/mean/count*
dtype0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized	eval_step*
dtype0*
_class
loc:@eval_step*
_output_shapes
: 
њ
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_12*
N*
T0
*
_output_shapes
:*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
Ё
$report_uninitialized_variables/ConstConst*
dtype0*Д
valueъBчBglobal_stepB7linear/text_ids_weighted_by_text_weights/weights/part_0Blinear/bias_weight/part_0B7read_batch_features/file_name_queue/limit_epochs/epochsBmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/countB	eval_step*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ў
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
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
х
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
б
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
ѓ
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
Ђ
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Ћ
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
_output_shapes
:*
T0*
Tshape0
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
џ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:€€€€€€€€€
ґ
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
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
Љ
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:€€€€€€€€€*
T0
°
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
ы
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized7linear/text_ids_weighted_by_text_weights/weights/part_0*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
њ
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedlinear/bias_weight/part_0*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
Ф
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_2*
N*
T0
*
_output_shapes
:*

axis 
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
ќ
&report_uninitialized_variables_1/ConstConst*
dtype0*t
valuekBiBglobal_stepB7linear/text_ids_weighted_by_text_weights/weights/part_0Blinear/bias_weight/part_0*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
г
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
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ы
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
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
≥
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
≥
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
—
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
_output_shapes
:*
T0*
Tshape0
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
б
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
Ю
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:€€€€€€€€€
Ї
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
И
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
м
init_2NoOp?^read_batch_features/file_name_queue/limit_epochs/epochs/Assign^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign^eval_step/Assign

init_all_tablesNoOp
/
group_deps_2NoOp^init_2^init_all_tables
£
Merge/MergeSummaryMergeSummary7read_batch_features/file_name_queue/fraction_of_32_full'read_batch_features/fraction_of_20_full_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_fulltraining_loss/ScalarSummary*
_output_shapes
: *
N
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_fc56d5d50f554169a72fef61b7ae0e26/part*
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
≤
save/SaveV2/tensor_namesConst*
dtype0*f
value]B[Bglobal_stepBlinear/bias_weightB0linear/text_ids_weighted_by_text_weights/weights*
_output_shapes
:
Г
save/SaveV2/shape_and_slicesConst*
dtype0*3
value*B(B B20 0,20B7179 20 0,7179:0,20*
_output_shapes
:
б
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_steplinear/bias_weight/part_0/read<linear/text_ids_weighted_by_text_weights/weights/part_0/read*
dtypes
2	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
Э
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
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2	*
_output_shapes
:
Ь
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
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
ј
save/Assign_1Assignlinear/bias_weight/part_0save/RestoreV2_1*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
Ц
save/RestoreV2_2/tensor_namesConst*
dtype0*E
value<B:B0linear/text_ids_weighted_by_text_weights/weights*
_output_shapes
:
}
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*(
valueBB7179 20 0,7179:0,20*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Б
save/Assign_2Assign7linear/text_ids_weighted_by_text_weights/weights/part_0save/RestoreV2_2*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	Л8
H
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2
-
save/restore_allNoOp^save/restore_shard"Жм∆
{     є)ќ	ШЅйу@÷AJо§
°5э4
9
Add
x"T
y"T
z"T"
Ttype:
2	
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
	summarizeintИ
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
p
	AssignAdd
ref"TА

value"T

output_ref"TА"
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
F
	CountUpTo
ref"TА
output"T"
limitint"
Ttype:
2	
A
Equal
x"T
y"T
z
"
Ttype:
2	
Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
Ѓ
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint€€€€€€€€€"
	containerstring "
shared_namestring И
4
Fill
dims

value"T
output"T"	
Ttype
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
М
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
ref"dtypeА
is_initialized
"
dtypetypeШ
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
Р


LogicalNot
x

y

К
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
2	Р

NoOp
М
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint€€€€€€€€€"	
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
п
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
К
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
Й
QueueDequeueUpToV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint€€€€€€€€€
~
QueueDequeueV2

handle

components2component_types"!
component_types
list(type)(0"

timeout_msint€€€€€€€€€
z
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint€€€€€€€€€
v
QueueEnqueueV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint€€€€€€€€€
#
QueueSizeV2

handle
size
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
Т
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
Љ
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ц
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
Й
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
compression_typestring И
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
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirtyВЈ

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
П
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
≤
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
У
)read_batch_features/file_name_queue/inputConst*
dtype0*µ
valueЂB®B.exout/features_test-00000-of-00006.tfrecord.gzB.exout/features_test-00001-of-00006.tfrecord.gzB.exout/features_test-00002-of-00006.tfrecord.gzB.exout/features_test-00003-of-00006.tfrecord.gzB.exout/features_test-00004-of-00006.tfrecord.gzB.exout/features_test-00005-of-00006.tfrecord.gz*
_output_shapes
:
j
(read_batch_features/file_name_queue/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
o
-read_batch_features/file_name_queue/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
∞
+read_batch_features/file_name_queue/GreaterGreater(read_batch_features/file_name_queue/Size-read_batch_features/file_name_queue/Greater/y*
T0*
_output_shapes
: 
І
0read_batch_features/file_name_queue/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
ѓ
8read_batch_features/file_name_queue/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
њ
1read_batch_features/file_name_queue/Assert/AssertAssert+read_batch_features/file_name_queue/Greater8read_batch_features/file_name_queue/Assert/Assert/data_0*
	summarize*

T
2
Љ
,read_batch_features/file_name_queue/IdentityIdentity)read_batch_features/file_name_queue/input2^read_batch_features/file_name_queue/Assert/Assert*
T0*
_output_shapes
:
x
6read_batch_features/file_name_queue/limit_epochs/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ы
7read_batch_features/file_name_queue/limit_epochs/epochs
VariableV2*
dtype0	*
shape: *
shared_name *
	container *
_output_shapes
: 
ѕ
>read_batch_features/file_name_queue/limit_epochs/epochs/AssignAssign7read_batch_features/file_name_queue/limit_epochs/epochs6read_batch_features/file_name_queue/limit_epochs/Const*
validate_shape(*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
use_locking(*
T0	*
_output_shapes
: 
о
<read_batch_features/file_name_queue/limit_epochs/epochs/readIdentity7read_batch_features/file_name_queue/limit_epochs/epochs*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
T0	*
_output_shapes
: 
ъ
:read_batch_features/file_name_queue/limit_epochs/CountUpTo	CountUpTo7read_batch_features/file_name_queue/limit_epochs/epochs*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
limit*
T0	*
_output_shapes
: 
ћ
0read_batch_features/file_name_queue/limit_epochsIdentity,read_batch_features/file_name_queue/Identity;^read_batch_features/file_name_queue/limit_epochs/CountUpTo*
T0*
_output_shapes
:
®
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
Ё
?read_batch_features/file_name_queue/file_name_queue_EnqueueManyQueueEnqueueManyV2#read_batch_features/file_name_queue0read_batch_features/file_name_queue/limit_epochs*

timeout_ms€€€€€€€€€*
Tcomponents
2
Н
9read_batch_features/file_name_queue/file_name_queue_CloseQueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues( 
П
;read_batch_features/file_name_queue/file_name_queue_Close_1QueueCloseV2#read_batch_features/file_name_queue*
cancel_pending_enqueues(
Д
8read_batch_features/file_name_queue/file_name_queue_SizeQueueSizeV2#read_batch_features/file_name_queue*
_output_shapes
: 
Ъ
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
§
'read_batch_features/file_name_queue/mulMul(read_batch_features/file_name_queue/Cast)read_batch_features/file_name_queue/mul/y*
T0*
_output_shapes
: 
і
<read_batch_features/file_name_queue/fraction_of_32_full/tagsConst*
dtype0*H
value?B= B7read_batch_features/file_name_queue/fraction_of_32_full*
_output_shapes
: 
–
7read_batch_features/file_name_queue/fraction_of_32_fullScalarSummary<read_batch_features/file_name_queue/fraction_of_32_full/tags'read_batch_features/file_name_queue/mul*
T0*
_output_shapes
: 
Х
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
ш
)read_batch_features/read/ReaderReadUpToV2ReaderReadUpToV2)read_batch_features/read/TFRecordReaderV2#read_batch_features/file_name_queue5read_batch_features/read/ReaderReadUpToV2/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
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
ю
+read_batch_features/read/ReaderReadUpToV2_1ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_1#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_1/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
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
ю
+read_batch_features/read/ReaderReadUpToV2_2ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_2#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_2/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
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
ю
+read_batch_features/read/ReaderReadUpToV2_3ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_3#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_3/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
[
read_batch_features/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
¶
read_batch_features/fifo_queueFIFOQueueV2*
capacity*
component_types
2*
_output_shapes
: *
shapes
: : *
	container *
shared_name 
В
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
Ў
6read_batch_features/cond/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue read_batch_features/cond/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
T0*
_output_shapes
: : 
К
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1Switch)read_batch_features/read/ReaderReadUpToV2 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
М
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2Switch+read_batch_features/read/ReaderReadUpToV2:1 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
©
/read_batch_features/cond/fifo_queue_EnqueueManyQueueEnqueueManyV28read_batch_features/cond/fifo_queue_EnqueueMany/Switch:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms€€€€€€€€€*
Tcomponents
2
г
+read_batch_features/cond/control_dependencyIdentity!read_batch_features/cond/switch_t0^read_batch_features/cond/fifo_queue_EnqueueMany*4
_class*
(&loc:@read_batch_features/cond/switch_t*
T0
*
_output_shapes
: 
I
read_batch_features/cond/NoOpNoOp"^read_batch_features/cond/switch_f
”
-read_batch_features/cond/control_dependency_1Identity!read_batch_features/cond/switch_f^read_batch_features/cond/NoOp*4
_class*
(&loc:@read_batch_features/cond/switch_f*
T0
*
_output_shapes
: 
ѓ
read_batch_features/cond/MergeMerge-read_batch_features/cond/control_dependency_1+read_batch_features/cond/control_dependency*
N*
T0
*
_output_shapes
: : 
Д
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
№
8read_batch_features/cond_1/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_1/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
T0*
_output_shapes
: : 
Т
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_1:1"read_batch_features/cond_1/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_1/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms€€€€€€€€€*
Tcomponents
2
л
-read_batch_features/cond_1/control_dependencyIdentity#read_batch_features/cond_1/switch_t2^read_batch_features/cond_1/fifo_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_1/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_1/NoOpNoOp$^read_batch_features/cond_1/switch_f
џ
/read_batch_features/cond_1/control_dependency_1Identity#read_batch_features/cond_1/switch_f ^read_batch_features/cond_1/NoOp*6
_class,
*(loc:@read_batch_features/cond_1/switch_f*
T0
*
_output_shapes
: 
µ
 read_batch_features/cond_1/MergeMerge/read_batch_features/cond_1/control_dependency_1-read_batch_features/cond_1/control_dependency*
N*
T0
*
_output_shapes
: : 
Д
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
№
8read_batch_features/cond_2/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_2/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
T0*
_output_shapes
: : 
Т
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_2"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_2:1"read_batch_features/cond_2/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_2/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms€€€€€€€€€*
Tcomponents
2
л
-read_batch_features/cond_2/control_dependencyIdentity#read_batch_features/cond_2/switch_t2^read_batch_features/cond_2/fifo_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_2/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_2/NoOpNoOp$^read_batch_features/cond_2/switch_f
џ
/read_batch_features/cond_2/control_dependency_1Identity#read_batch_features/cond_2/switch_f ^read_batch_features/cond_2/NoOp*6
_class,
*(loc:@read_batch_features/cond_2/switch_f*
T0
*
_output_shapes
: 
µ
 read_batch_features/cond_2/MergeMerge/read_batch_features/cond_2/control_dependency_1-read_batch_features/cond_2/control_dependency*
N*
T0
*
_output_shapes
: : 
Д
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
№
8read_batch_features/cond_3/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_3/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
T0*
_output_shapes
: : 
Т
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_3"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_3:1"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_3/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms€€€€€€€€€*
Tcomponents
2
л
-read_batch_features/cond_3/control_dependencyIdentity#read_batch_features/cond_3/switch_t2^read_batch_features/cond_3/fifo_queue_EnqueueMany*6
_class,
*(loc:@read_batch_features/cond_3/switch_t*
T0
*
_output_shapes
: 
M
read_batch_features/cond_3/NoOpNoOp$^read_batch_features/cond_3/switch_f
џ
/read_batch_features/cond_3/control_dependency_1Identity#read_batch_features/cond_3/switch_f ^read_batch_features/cond_3/NoOp*6
_class,
*(loc:@read_batch_features/cond_3/switch_f*
T0
*
_output_shapes
: 
µ
 read_batch_features/cond_3/MergeMerge/read_batch_features/cond_3/control_dependency_1-read_batch_features/cond_3/control_dependency*
N*
T0
*
_output_shapes
: : 
s
$read_batch_features/fifo_queue_CloseQueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues( 
u
&read_batch_features/fifo_queue_Close_1QueueCloseV2read_batch_features/fifo_queue*
cancel_pending_enqueues(
j
#read_batch_features/fifo_queue_SizeQueueSizeV2read_batch_features/fifo_queue*
_output_shapes
: 
u
read_batch_features/CastCast#read_batch_features/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
^
read_batch_features/mul/yConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
t
read_batch_features/mulMulread_batch_features/Castread_batch_features/mul/y*
T0*
_output_shapes
: 
Ф
,read_batch_features/fraction_of_20_full/tagsConst*
dtype0*8
value/B- B'read_batch_features/fraction_of_20_full*
_output_shapes
: 
†
'read_batch_features/fraction_of_20_fullScalarSummary,read_batch_features/fraction_of_20_full/tagsread_batch_features/mul*
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
 
read_batch_featuresQueueDequeueUpToV2read_batch_features/fifo_queueread_batch_features/n*

timeout_ms€€€€€€€€€*
component_types
2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
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
Љ
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
Д
;read_batch_features/ParseExample/ParseExample/sparse_keys_0Const*
dtype0*
valueB Btext_ids*
_output_shapes
: 
И
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
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_1Const*
dtype0*
valueB Btarget*
_output_shapes
: 
±
-read_batch_features/ParseExample/ParseExampleParseExampleread_batch_features:13read_batch_features/ParseExample/ParseExample/names;read_batch_features/ParseExample/ParseExample/sparse_keys_0;read_batch_features/ParseExample/ParseExample/sparse_keys_1:read_batch_features/ParseExample/ParseExample/dense_keys_0:read_batch_features/ParseExample/ParseExample/dense_keys_1(read_batch_features/ParseExample/Reshape&read_batch_features/ParseExample/Const*
dense_shapes
: : *В
_output_shapesp
n:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::€€€€€€€€€:€€€€€€€€€*
Ndense*
sparse_types
2	*
Tdense
2		*
Nsparse
Ђ
 read_batch_features/fifo_queue_1FIFOQueueV2*
capacityd* 
component_types
2								*
_output_shapes
: *
shapes
 *
	container *
shared_name 
n
%read_batch_features/fifo_queue_1_SizeQueueSizeV2 read_batch_features/fifo_queue_1*
_output_shapes
: 
y
read_batch_features/Cast_1Cast%read_batch_features/fifo_queue_1_Size*

DstT0*

SrcT0*
_output_shapes
: 
`
read_batch_features/mul_1/yConst*
dtype0*
valueB
 *
„#<*
_output_shapes
: 
z
read_batch_features/mul_1Mulread_batch_features/Cast_1read_batch_features/mul_1/y*
T0*
_output_shapes
: 
Д
dread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full/tagsConst*
dtype0*p
valuegBe B_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full*
_output_shapes
: 
Т
_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_fullScalarSummarydread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full/tagsread_batch_features/mul_1*
T0*
_output_shapes
: 
∞
(read_batch_features/fifo_queue_1_enqueueQueueEnqueueV2 read_batch_features/fifo_queue_1/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms€€€€€€€€€*
Tcomponents
2								
≤
*read_batch_features/fifo_queue_1_enqueue_1QueueEnqueueV2 read_batch_features/fifo_queue_1/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:5read_batch_features*

timeout_ms€€€€€€€€€*
Tcomponents
2								
w
&read_batch_features/fifo_queue_1_CloseQueueCloseV2 read_batch_features/fifo_queue_1*
cancel_pending_enqueues( 
y
(read_batch_features/fifo_queue_1_Close_1QueueCloseV2 read_batch_features/fifo_queue_1*
cancel_pending_enqueues(
≠
(read_batch_features/fifo_queue_1_DequeueQueueDequeueV2 read_batch_features/fifo_queue_1*

timeout_ms€€€€€€€€€* 
component_types
2								*С
_output_shapes
}:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€::€€€€€€€€€:€€€€€€€€€::€€€€€€€€€
Y
ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Р

ExpandDims
ExpandDims(read_batch_features/fifo_queue_1_DequeueExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:€€€€€€€€€
[
ExpandDims_1/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Ц
ExpandDims_1
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:1ExpandDims_1/dim*

Tdim0*
T0	*'
_output_shapes
:€€€€€€€€€
V
linear/linear/mod/yConst*
dtype0	*
value
B	 RЛ8*
_output_shapes
: 
М
linear/linear/modFloorMod*read_batch_features/fifo_queue_1_Dequeue:3linear/linear/mod/y*
T0	*#
_output_shapes
:€€€€€€€€€
м
Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/ConstConst*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
valueB	Л8*    *
_output_shapes
:	Л8
щ
7linear/text_ids_weighted_by_text_weights/weights/part_0
VariableV2*
	container *
_output_shapes
:	Л8*
dtype0*
shape:	Л8*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
shared_name 
л
>linear/text_ids_weighted_by_text_weights/weights/part_0/AssignAssign7linear/text_ids_weighted_by_text_weights/weights/part_0Ilinear/text_ids_weighted_by_text_weights/weights/part_0/Initializer/Const*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	Л8
ч
<linear/text_ids_weighted_by_text_weights/weights/part_0/readIdentity7linear/text_ids_weighted_by_text_weights/weights/part_0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
T0*
_output_shapes
:	Л8
ѓ
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
Ѓ
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
У
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SliceSlice*read_batch_features/fifo_queue_1_Dequeue:4elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/begindlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
©
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
ж
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ProdProd_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
™
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
ѕ
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GatherGather*read_batch_features/fifo_queue_1_Dequeue:4hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
х
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shapePack^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Prod`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather*
_output_shapes
:*

axis *
T0	*
N
т
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeSparseReshape*read_batch_features/fifo_queue_1_Dequeue:2*read_batch_features/fifo_queue_1_Dequeue:4qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/new_shape*-
_output_shapes
:€€€€€€€€€:
љ
plinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/IdentityIdentitylinear/linear/mod*
T0	*#
_output_shapes
:€€€€€€€€€
™
hlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
А
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualGreaterEqualplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityhlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
®
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Ђ
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterGreater*read_batch_features/fifo_queue_1_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater/y*
T0*#
_output_shapes
:€€€€€€€€€
в
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd
LogicalAndflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/GreaterEqualalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Greater*#
_output_shapes
:€€€€€€€€€
ч
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/WhereWheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:€€€€€€€€€
Ї
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
т
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ReshapeReshape_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Whereglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:€€€€€€€€€
Ш
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:€€€€€€€€€
Э
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2Gatherplinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape/Identityalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:€€€€€€€€€
ю
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityIdentityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
щ
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1Wheredlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/LogicalAnd*'
_output_shapes
:€€€€€€€€€
Љ
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
ш
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1Reshapealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Where_1ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1/shape*
Tshape0*
T0	*#
_output_shapes
:€€€€€€€€€
Ъ
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3Gatherglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:€€€€€€€€€
ў
blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4Gather*read_batch_features/fifo_queue_1_Dequeue:6clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_1*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
А
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Identityilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseReshape:1*
T0	*
_output_shapes
:
µ
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
ћ
Бlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
ќ
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
ќ
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
С
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_sliceStridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityБlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stackГlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_1Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
І
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/CastCast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
ї
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
ї
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/rangeRangeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/startrlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Castylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
Ѓ
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1Castslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:€€€€€€€€€
’
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
„
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
„
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
¶
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stackЕlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_1Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
√
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffListDifftlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Cast_1}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Щ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/IdentityГlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stackЕlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_1Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
«
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
і
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims
ExpandDims}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/strided_slice_2|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
ћ
Йlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
ћ
Йlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
м
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseSparseToDensevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiffxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ExpandDimsЙlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/sparse_valuesЙlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:€€€€€€€€€
ћ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
µ
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ReshapeReshapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:€€€€€€€€€
Ѓ
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_like	ZerosLikeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:€€€€€€€€€
ї
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
≥
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concatConcatV2ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Reshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/zeros_likeylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat/axis*'
_output_shapes
:€€€€€€€€€*

Tidx0*
T0	*
N
©
slinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ShapeShapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
Т
rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/FillFillslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Shapeslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:€€€€€€€€€
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
†
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_1tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:€€€€€€€€€*

Tidx0*
T0	*
N
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ъ
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_2rlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/Fill{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:€€€€€€€€€*

Tidx0*
T0	*
N
°
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorderSparseReordervlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_1vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/concat_2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
Л
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/IdentityIdentityblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity*
T0	*
_output_shapes
:
Ї
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
ќ
Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ы
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_sliceStridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Гlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stackЕlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_1Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
Ђ
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/CastCast}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Я
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/rangeRange{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/starttlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
≤
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1Castulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/range*

DstT0	*

SrcT0*#
_output_shapes
:€€€€€€€€€
„
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
ў
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
ў
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ѓ
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1StridedSliceblinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stackЗlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_1Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
…
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffListDiffvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Cast_1linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
–
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
“
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
“
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
£
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2StridedSlicedlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stackЗlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_1Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
…
~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Ї
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims
ExpandDimslinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/strided_slice_2~linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
ќ
Лlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
ќ
Лlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
ц
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDenseSparseToDensexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiffzlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ExpandDimsЛlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesЛlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:€€€€€€€€€
ќ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
ї
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ReshapeReshapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:€€€€€€€€€
≤
zlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like	ZerosLikewlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshape*
T0	*'
_output_shapes
:€€€€€€€€€
љ
{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
ї
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Reshapezlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/zeros_like{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat/axis*'
_output_shapes
:€€€€€€€€€*

Tidx0*
T0	*
N
≠
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ShapeShapexlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/ListDiff*
out_type0*
T0	*
_output_shapes
:
Ш
tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/FillFillulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Shapeulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Const*
T0*#
_output_shapes
:€€€€€€€€€
њ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
¶
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_3vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1/axis*'
_output_shapes
:€€€€€€€€€*

Tidx0*
T0	*
N
њ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
†
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2ConcatV2blinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Gather_4tlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2/axis*#
_output_shapes
:€€€€€€€€€*

Tidx0*
T0*
N
©
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorderSparseReorderxlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/concat_2dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
П
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/IdentityIdentitydlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Identity_1*
T0	*
_output_shapes
:
„
Еlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
ў
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
ў
Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
«
linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_sliceStridedSlice{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorderЕlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stackЗlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_1Зlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
Љ
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/CastCastlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€
ц
Вlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupGather<linear/text_ids_weighted_by_text_weights/weights/part_0/read}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseReorder:1*
validate_indices(*
Tparams0*
Tindices0	*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*'
_output_shapes
:€€€€€€€€€
Є
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/RankConst*
dtype0*
value	B :*
_output_shapes
: 
є
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
О
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/subSubvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Rankwlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/sub/y*
T0*
_output_shapes
: 
√
Аlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
µ
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims
ExpandDimsulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/subАlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Њ
|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
®
vlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/FillFill|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ExpandDims|linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill/value*
T0*#
_output_shapes
:€€€€€€€€€
ґ
wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ShapeShapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1*
out_type0*
T0*
_output_shapes
:
њ
}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ј
xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concatConcatV2wlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Shapevlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Fill}linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat/axis*#
_output_shapes
:€€€€€€€€€*

Tidx0*
T0*
N
њ
ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/ReshapeReshapelinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows_1/SparseReorder:1xlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/concat*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
Ѓ
ulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulMulВlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/embedding_lookupylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:€€€€€€€€€
∞
qlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse
SegmentSumulinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/mulvlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:€€€€€€€€€
Ї
ilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
Ц
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2Reshape{linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/SparseFillEmptyRows/SparseToDenseilinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2/shape*
Tshape0*
T0
*'
_output_shapes
:€€€€€€€€€
Р
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/ShapeShapeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Ј
mlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
є
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
є
olinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ї
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
£
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
н
_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stackPackalinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack/0glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/strided_slice*
_output_shapes
:*

axis *
T0*
N
щ
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/TileTileclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_2_linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ц
dlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_like	ZerosLikeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
ќ
Ylinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weightsSelect^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Tiledlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/zeros_likeqlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
∆
^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/CastCast*read_batch_features/fifo_queue_1_Dequeue:4*

DstT0*

SrcT0	*
_output_shapes
:
±
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
∞
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Ќ
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1Slice^linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Castglinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ъ
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1ShapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights*
out_type0*
T0*
_output_shapes
:
±
glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
є
flinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/sizeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
–
alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2Slicealinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Shape_1glinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/beginflinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
І
elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
”
`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concatConcatV2alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_1alinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Slice_2elinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
л
clinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3ReshapeYlinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights`linear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/concat*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
l
linear/linear/Reshape/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
в
linear/linear/ReshapeReshapeclinear/linear/text_ids_weighted_by_text_weights/text_ids_weighted_by_text_weights_weights/Reshape_3linear/linear/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
¶
+linear/bias_weight/part_0/Initializer/ConstConst*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
valueB*    *
_output_shapes
:
≥
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
о
 linear/bias_weight/part_0/AssignAssignlinear/bias_weight/part_0+linear/bias_weight/part_0/Initializer/Const*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
Ш
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
Ф
linear/linear/BiasAddBiasAddlinear/linear/Reshapelinear/bias_weight*
data_formatNHWC*
T0*'
_output_shapes
:€€€€€€€€€
m
predictions/probabilitiesSoftmaxlinear/linear/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
_
predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
Н
predictions/classesArgMaxlinear/linear/BiasAddpredictions/classes/dimension*

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
О
0training_loss/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
.training_loss/softmax_cross_entropy_loss/ShapeShape0training_loss/softmax_cross_entropy_loss/Squeeze*
out_type0*
T0	*
_output_shapes
:
и
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
]
training_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Т
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
У
,metrics/remove_squeezable_dimensions/SqueezeSqueezeExpandDims_1*
squeeze_dims

€€€€€€€€€*
T0	*#
_output_shapes
:€€€€€€€€€
З
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:€€€€€€€€€
c
metrics/ToFloatCastmetrics/Equal*

DstT0*

SrcT0
*#
_output_shapes
:€€€€€€€€€
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
ћ
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
validate_shape(*)
_class
loc:@metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
Л
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
ќ
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
validate_shape(*)
_class
loc:@metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
Л
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
T0*
_output_shapes
: 
_
metrics/accuracy/SizeSizemetrics/ToFloat*
out_type0*
T0*
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
В
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
і
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*)
_class
loc:@metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
Љ
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
П
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
В
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
А
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
Ы
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
Т
metrics/Assert/ConstConst*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
Ъ
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
А
metrics/Reshape/shapeConst^metrics/Assert/Assert*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
{
metrics/ReshapeReshapeExpandDims_1metrics/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:€€€€€€€€€
]
metrics/one_hot/on_valueConst*
dtype0*
valueB
 *  А?*
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
«
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
TI0	*'
_output_shapes
:€€€€€€€€€*
T0*
axis€€€€€€€€€
f
metrics/CastCastmetrics/one_hot*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€
j
metrics/auc/Reshape/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
Ф
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"   €€€€*
_output_shapes
:
Л
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:€€€€€€€€€
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
µ
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
А
metrics/auc/ConstConst*
dtype0*є
valueѓBђ»"†Хњ÷≥ѕ©§;ѕ©$<Јюv<ѕ©§<C‘Ќ<Јюц<Х=ѕ©$=	?9=C‘M=}ib=Јюv=ш…Е=ХР=2_Ъ=ѕ©§=lфЃ=	?є=¶Й√=C‘Ќ=аЎ=}iв=ім=Јюц=™§ >ш…>Gп
>Х>д9>2_>БД>ѕ©$>ѕ)>lф.>ї4>	?9>Wd>>¶ЙC>фЃH>C‘M>СщR>аX>.D]>}ib>ЋОg>іl>hўq>Јюv>$|>™§А>Q7Г>ш…Е>†\И>GпК>оБН>ХР><ІТ>д9Х>ЛћЧ>2_Ъ>ўсЬ>БДЯ>(Ґ>ѕ©§>v<І>ѕ©>≈aђ>lфЃ>З±>їі>bђґ>	?є>∞—ї>WdЊ>€цј>¶Й√>M∆>фЃ»>ЬAЋ>C‘Ќ>кf–>Сщ“>9М’>аЎ>З±Џ>.DЁ>÷÷я>}iв>$ьд>ЋОз>r!к>ім>ЅFп>hўс>lф>Јюц>^Сщ>$ь>ђґю>™§ ?эн?Q7?•А?ш…?L?†\?у•	?Gп
?Ъ8?оБ?BЋ?Х?й]?<І?Рр?д9?7Г?Лћ?я?2_?Ж®?ўс?-;?БД?‘Ќ ?("?{`#?ѕ©$?#у%?v<'? Е(?ѕ)?q+?≈a,?Ђ-?lф.?ј=0?З1?g–2?ї4?c5?bђ6?µх7?	?9?]И:?∞—;?=?Wd>?Ђ≠??€ц@?R@B?¶ЙC?ъ“D?MF?°eG?фЃH?HшI?ЬAK?пКL?C‘M?ЧO?кfP?>∞Q?СщR?еBT?9МU?М’V?аX?3hY?З±Z?џъ[?.D]?ВН^?÷÷_?) a?}ib?–≤c?$ьd?xEf?ЋОg?Ўh?r!j?∆jk?іl?mэm?ЅFo?Рp?hўq?Љ"s?lt?cµu?Јюv?
Hx?^Сy?≤Џz?$|?Ym}?ђґ~? А?*
_output_shapes	
:»
d
metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
Й
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	»
U
metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
Г
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
_output_shapes
:*

axis *
T0*
N
И
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:»€€€€€€€€€
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
Ѓ
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
T0*
_output_shapes
:
У
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:€€€€€€€€€
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"»      *
_output_shapes
:
Ф
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:»€€€€€€€€€
w
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*(
_output_shapes
:»€€€€€€€€€
c
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"»      *
_output_shapes
:
Ф
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:»€€€€€€€€€
d
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2*(
_output_shapes
:»€€€€€€€€€
`
metrics/auc/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»
И
metrics/auc/true_positives
VariableV2*
dtype0*
shape:»*
shared_name *
	container *
_output_shapes	
:»
Ў
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:»
Ь
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
T0*
_output_shapes	
:»
w
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
w
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

DstT0*

SrcT0
*(
_output_shapes
:»€€€€€€€€€
c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
У
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:»
Ј
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*-
_class#
!loc:@metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:»
b
metrics/auc/zeros_1Const*
dtype0*
valueB»*    *
_output_shapes	
:»
Й
metrics/auc/false_negatives
VariableV2*
dtype0*
shape:»*
shared_name *
	container *
_output_shapes	
:»
Ё
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
validate_shape(*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:»
Я
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
_output_shapes	
:»
|
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

DstT0*

SrcT0
*(
_output_shapes
:»€€€€€€€€€
e
#metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:»
љ
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:»
b
metrics/auc/zeros_2Const*
dtype0*
valueB»*    *
_output_shapes	
:»
И
metrics/auc/true_negatives
VariableV2*
dtype0*
shape:»*
shared_name *
	container *
_output_shapes	
:»
Џ
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:»
Ь
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
_output_shapes	
:»
В
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

DstT0*

SrcT0
*(
_output_shapes
:»€€€€€€€€€
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:»
ї
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:»
b
metrics/auc/zeros_3Const*
dtype0*
valueB»*    *
_output_shapes	
:»
Й
metrics/auc/false_positives
VariableV2*
dtype0*
shape:»*
shared_name *
	container *
_output_shapes	
:»
Ё
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:»
Я
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
T0*
_output_shapes	
:»

metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

DstT0*

SrcT0
*(
_output_shapes
:»€€€€€€€€€
e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:»
љ
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*.
_class$
" loc:@metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:»
V
metrics/auc/add/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:»
Б
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:»
X
metrics/auc/add_2/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
T0*
_output_shapes	
:»
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
T0*
_output_shapes	
:»
Б
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:»
X
metrics/auc/add_4/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:»
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:»
k
!metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:«*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
¬
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

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
¬
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:«
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:«*
_output_shapes
:
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ј
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

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
ј
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:«
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
:«
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:«
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
 *љ7Ж5*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
T0*
_output_shapes	
:»
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:»
X
metrics/auc/add_8/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:»
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
T0*
_output_shapes	
:»
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:»
Y
metrics/auc/add_10/yConst*
dtype0*
valueB
 *љ7Ж5*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:»
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
T0*
_output_shapes	
:»
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:«*
_output_shapes
:
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
¬
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

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
¬
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
T0*
_output_shapes	
:«
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:«*
_output_shapes
:
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
¬
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

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
¬
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:«*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
T0*
_output_shapes	
:«
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
:«
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:«
]
metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
В
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
И
*metrics/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_1*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
Т
(metrics/softmax_cross_entropy_loss/ShapeShape*metrics/softmax_cross_entropy_loss/Squeeze*
out_type0*
T0	*
_output_shapes
:
№
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitslinear/linear/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
a
metrics/eval_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ф
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
Љ
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
Њ
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
§
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
ђ
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
Л
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
`

group_depsNoOp^metrics/mean/update_op^metrics/auc/update_op^metrics/accuracy/update_op
\
eval_step/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
m
	eval_step
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
¶
eval_step/AssignAssign	eval_stepeval_step/initial_value*
validate_shape(*
_class
loc:@eval_step*
use_locking(*
T0*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
T0*
_output_shapes
: 
T
AssignAdd/valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
use_locking( *
T0*
_output_shapes
: 
Е
initNoOp^global_step/Assign?^linear/text_ids_weighted_by_text_weights/weights/part_0/Assign!^linear/bias_weight/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
щ
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized7linear/text_ids_weighted_by_text_weights/weights/part_0*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
љ
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedlinear/bias_weight/part_0*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
щ
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized7read_batch_features/file_name_queue/limit_epochs/epochs*
dtype0	*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
_output_shapes
: 
Ј
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedmetrics/accuracy/total*
dtype0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
Ј
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedmetrics/accuracy/count*
dtype0*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: 
њ
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedmetrics/auc/true_positives*
dtype0*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes
: 
Ѕ
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedmetrics/auc/false_negatives*
dtype0*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes
: 
њ
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializedmetrics/auc/true_negatives*
dtype0*-
_class#
!loc:@metrics/auc/true_negatives*
_output_shapes
: 
Ѕ
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedmetrics/auc/false_positives*
dtype0*.
_class$
" loc:@metrics/auc/false_positives*
_output_shapes
: 
∞
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedmetrics/mean/total*
dtype0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
∞
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedmetrics/mean/count*
dtype0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized	eval_step*
dtype0*
_class
loc:@eval_step*
_output_shapes
: 
њ
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_12*
_output_shapes
:*

axis *
T0
*
N
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
Ё
$report_uninitialized_variables/ConstConst*
dtype0*Д
valueъBчBglobal_stepB7linear/text_ids_weighted_by_text_weights/weights/part_0Blinear/bias_weight/part_0B7read_batch_features/file_name_queue/limit_epochs/epochsBmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/countB	eval_step*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ў
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
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
х
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
б
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
ѓ
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
Ђ
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
Ћ
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
џ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:
Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:€€€€€€€€€
ґ
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
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
Љ
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*#
_output_shapes
:€€€€€€€€€*

Tidx0*
T0*
N
°
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
ы
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized7linear/text_ids_weighted_by_text_weights/weights/part_0*
dtype0*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
_output_shapes
: 
њ
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedlinear/bias_weight/part_0*
dtype0*,
_class"
 loc:@linear/bias_weight/part_0*
_output_shapes
: 
Ф
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_2*
_output_shapes
:*

axis *
T0
*
N
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
ќ
&report_uninitialized_variables_1/ConstConst*
dtype0*t
valuekBiBglobal_stepB7linear/text_ids_weighted_by_text_weights/weights/part_0Blinear/bias_weight/part_0*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
г
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
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ы
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
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
≥
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
≥
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
—
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
б
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:
Ю
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:€€€€€€€€€
Ї
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
И
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
м
init_2NoOp?^read_batch_features/file_name_queue/limit_epochs/epochs/Assign^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign^eval_step/Assign

init_all_tablesNoOp
/
group_deps_2NoOp^init_2^init_all_tables
£
Merge/MergeSummaryMergeSummary7read_batch_features/file_name_queue/fraction_of_32_full'read_batch_features/fraction_of_20_full_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_fulltraining_loss/ScalarSummary*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_fc56d5d50f554169a72fef61b7ae0e26/part*
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
≤
save/SaveV2/tensor_namesConst*
dtype0*f
value]B[Bglobal_stepBlinear/bias_weightB0linear/text_ids_weighted_by_text_weights/weights*
_output_shapes
:
Г
save/SaveV2/shape_and_slicesConst*
dtype0*3
value*B(B B20 0,20B7179 20 0,7179:0,20*
_output_shapes
:
б
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_steplinear/bias_weight/part_0/read<linear/text_ids_weighted_by_text_weights/weights/part_0/read*
dtypes
2	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
Э
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
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2	*
_output_shapes
:
Ь
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
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
ј
save/Assign_1Assignlinear/bias_weight/part_0save/RestoreV2_1*
validate_shape(*,
_class"
 loc:@linear/bias_weight/part_0*
use_locking(*
T0*
_output_shapes
:
Ц
save/RestoreV2_2/tensor_namesConst*
dtype0*E
value<B:B0linear/text_ids_weighted_by_text_weights/weights*
_output_shapes
:
}
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*(
valueBB7179 20 0,7179:0,20*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Б
save/Assign_2Assign7linear/text_ids_weighted_by_text_weights/weights/part_0save/RestoreV2_2*
validate_shape(*J
_class@
><loc:@linear/text_ids_weighted_by_text_weights/weights/part_0*
use_locking(*
T0*
_output_shapes
:	Л8
H
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2
-
save/restore_allNoOp^save/restore_shard"" 
global_step

global_step:0"Ю
trainable_variablesЖГ
э
9linear/text_ids_weighted_by_text_weights/weights/part_0:0>linear/text_ids_weighted_by_text_weights/weights/part_0/Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/read:0"@
0linear/text_ids_weighted_by_text_weights/weightsЛ8  "Л8
А
linear/bias_weight/part_0:0 linear/bias_weight/part_0/Assign linear/bias_weight/part_0/read:0"
linear/bias_weight ""!
local_init_op

group_deps_2"Ќ
	variablesњЉ
7
global_step:0global_step/Assignglobal_step/read:0
э
9linear/text_ids_weighted_by_text_weights/weights/part_0:0>linear/text_ids_weighted_by_text_weights/weights/part_0/Assign>linear/text_ids_weighted_by_text_weights/weights/part_0/read:0"@
0linear/text_ids_weighted_by_text_weights/weightsЛ8  "Л8
А
linear/bias_weight/part_0:0 linear/bias_weight/part_0/Assign linear/bias_weight/part_0/read:0"
linear/bias_weight ""U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"щ
	summariesл
и
9read_batch_features/file_name_queue/fraction_of_32_full:0
)read_batch_features/fraction_of_20_full:0
aread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full:0
training_loss/ScalarSummary:0"П#
cond_contextю"ы"
ћ
"read_batch_features/cond/cond_text"read_batch_features/cond/pred_id:0#read_batch_features/cond/switch_t:0 *Џ
-read_batch_features/cond/control_dependency:0
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch:1
:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1:1
:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2:1
"read_batch_features/cond/pred_id:0
#read_batch_features/cond/switch_t:0
 read_batch_features/fifo_queue:0
+read_batch_features/read/ReaderReadUpToV2:0
+read_batch_features/read/ReaderReadUpToV2:1\
 read_batch_features/fifo_queue:08read_batch_features/cond/fifo_queue_EnqueueMany/Switch:1i
+read_batch_features/read/ReaderReadUpToV2:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2:1i
+read_batch_features/read/ReaderReadUpToV2:0:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1:1
л
$read_batch_features/cond/cond_text_1"read_batch_features/cond/pred_id:0#read_batch_features/cond/switch_f:0*z
/read_batch_features/cond/control_dependency_1:0
"read_batch_features/cond/pred_id:0
#read_batch_features/cond/switch_f:0
м
$read_batch_features/cond_1/cond_text$read_batch_features/cond_1/pred_id:0%read_batch_features/cond_1/switch_t:0 *ф
/read_batch_features/cond_1/control_dependency:0
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch:1
<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1:1
<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2:1
$read_batch_features/cond_1/pred_id:0
%read_batch_features/cond_1/switch_t:0
 read_batch_features/fifo_queue:0
-read_batch_features/read/ReaderReadUpToV2_1:0
-read_batch_features/read/ReaderReadUpToV2_1:1^
 read_batch_features/fifo_queue:0:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch:1m
-read_batch_features/read/ReaderReadUpToV2_1:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2:1m
-read_batch_features/read/ReaderReadUpToV2_1:0<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1:1
ш
&read_batch_features/cond_1/cond_text_1$read_batch_features/cond_1/pred_id:0%read_batch_features/cond_1/switch_f:0*А
1read_batch_features/cond_1/control_dependency_1:0
$read_batch_features/cond_1/pred_id:0
%read_batch_features/cond_1/switch_f:0
м
$read_batch_features/cond_2/cond_text$read_batch_features/cond_2/pred_id:0%read_batch_features/cond_2/switch_t:0 *ф
/read_batch_features/cond_2/control_dependency:0
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch:1
<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1:1
<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2:1
$read_batch_features/cond_2/pred_id:0
%read_batch_features/cond_2/switch_t:0
 read_batch_features/fifo_queue:0
-read_batch_features/read/ReaderReadUpToV2_2:0
-read_batch_features/read/ReaderReadUpToV2_2:1^
 read_batch_features/fifo_queue:0:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch:1m
-read_batch_features/read/ReaderReadUpToV2_2:0<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1:1m
-read_batch_features/read/ReaderReadUpToV2_2:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2:1
ш
&read_batch_features/cond_2/cond_text_1$read_batch_features/cond_2/pred_id:0%read_batch_features/cond_2/switch_f:0*А
1read_batch_features/cond_2/control_dependency_1:0
$read_batch_features/cond_2/pred_id:0
%read_batch_features/cond_2/switch_f:0
м
$read_batch_features/cond_3/cond_text$read_batch_features/cond_3/pred_id:0%read_batch_features/cond_3/switch_t:0 *ф
/read_batch_features/cond_3/control_dependency:0
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch:1
<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1:1
<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2:1
$read_batch_features/cond_3/pred_id:0
%read_batch_features/cond_3/switch_t:0
 read_batch_features/fifo_queue:0
-read_batch_features/read/ReaderReadUpToV2_3:0
-read_batch_features/read/ReaderReadUpToV2_3:1^
 read_batch_features/fifo_queue:0:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch:1m
-read_batch_features/read/ReaderReadUpToV2_3:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2:1m
-read_batch_features/read/ReaderReadUpToV2_3:0<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1:1
ш
&read_batch_features/cond_3/cond_text_1$read_batch_features/cond_3/pred_id:0%read_batch_features/cond_3/switch_f:0*А
1read_batch_features/cond_3/control_dependency_1:0
$read_batch_features/cond_3/pred_id:0
%read_batch_features/cond_3/switch_f:0"є
local_variables•
Ґ
9read_batch_features/file_name_queue/limit_epochs/epochs:0
metrics/accuracy/total:0
metrics/accuracy/count:0
metrics/auc/true_positives:0
metrics/auc/false_negatives:0
metrics/auc/true_negatives:0
metrics/auc/false_positives:0
metrics/mean/total:0
metrics/mean/count:0
eval_step:0"d
linearZ
X
9linear/text_ids_weighted_by_text_weights/weights/part_0:0
linear/bias_weight/part_0:0"ћ
queue_runnersЇЈ
б
#read_batch_features/file_name_queue?read_batch_features/file_name_queue/file_name_queue_EnqueueMany9read_batch_features/file_name_queue/file_name_queue_Close";read_batch_features/file_name_queue/file_name_queue_Close_1*
€
read_batch_features/fifo_queue read_batch_features/cond/Merge:0"read_batch_features/cond_1/Merge:0"read_batch_features/cond_2/Merge:0"read_batch_features/cond_3/Merge:0$read_batch_features/fifo_queue_Close"&read_batch_features/fifo_queue_Close_1*
ќ
 read_batch_features/fifo_queue_1(read_batch_features/fifo_queue_1_enqueue*read_batch_features/fifo_queue_1_enqueue_1&read_batch_features/fifo_queue_1_Close"(read_batch_features/fifo_queue_1_Close_1*"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"&

summary_op

Merge/MergeSummary:0"
	eval_step

eval_step:0"
ready_op


concat:0"m
model_variablesZ
X
9linear/text_ids_weighted_by_text_weights/weights/part_0:0
linear/bias_weight/part_0:0"
init_op

group_deps_1<СF5G       Ї√ыЫ	йу@÷AРN*9

lossАе>


auc%}?

global_step

accuracy7Йa?м`≥[