       £K"	   K°9÷Abrain.Event:2ърЗсЇ     R§4S	HМ+K°9÷A"Рф

global_step/Initializer/ConstConst*
_output_shapes
: *
dtype0	*
_class
loc:@global_step*
value	B	 R 
П
global_step
VariableV2*
shape: *
_output_shapes
: *
shared_name *
_class
loc:@global_step*
dtype0	*
	container 
≤
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
use_locking(*
validate_shape(*
T0	*
_output_shapes
: *
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_output_shapes
: *
_class
loc:@global_step
¶
)read_batch_features/file_name_queue/inputConst*I
value@B>B4../tfpreout/features_eval-00000-of-00001.tfrecord.gz*
dtype0*
_output_shapes
:
j
(read_batch_features/file_name_queue/SizeConst*
value	B :*
_output_shapes
: *
dtype0
o
-read_batch_features/file_name_queue/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 
∞
+read_batch_features/file_name_queue/GreaterGreater(read_batch_features/file_name_queue/Size-read_batch_features/file_name_queue/Greater/y*
_output_shapes
: *
T0
І
0read_batch_features/file_name_queue/Assert/ConstConst*G
value>B< B6string_input_producer requires a non-null input tensor*
dtype0*
_output_shapes
: 
ѓ
8read_batch_features/file_name_queue/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor
њ
1read_batch_features/file_name_queue/Assert/AssertAssert+read_batch_features/file_name_queue/Greater8read_batch_features/file_name_queue/Assert/Assert/data_0*

T
2*
	summarize
Љ
,read_batch_features/file_name_queue/IdentityIdentity)read_batch_features/file_name_queue/input2^read_batch_features/file_name_queue/Assert/Assert*
T0*
_output_shapes
:
x
6read_batch_features/file_name_queue/limit_epochs/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Ы
7read_batch_features/file_name_queue/limit_epochs/epochs
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
ѕ
>read_batch_features/file_name_queue/limit_epochs/epochs/AssignAssign7read_batch_features/file_name_queue/limit_epochs/epochs6read_batch_features/file_name_queue/limit_epochs/Const*
use_locking(*
T0	*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
validate_shape(*
_output_shapes
: 
о
<read_batch_features/file_name_queue/limit_epochs/epochs/readIdentity7read_batch_features/file_name_queue/limit_epochs/epochs*
T0	*
_output_shapes
: *J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs
ъ
:read_batch_features/file_name_queue/limit_epochs/CountUpTo	CountUpTo7read_batch_features/file_name_queue/limit_epochs/epochs*
T0	*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
_output_shapes
: *
limit
ћ
0read_batch_features/file_name_queue/limit_epochsIdentity,read_batch_features/file_name_queue/Identity;^read_batch_features/file_name_queue/limit_epochs/CountUpTo*
T0*
_output_shapes
:
®
#read_batch_features/file_name_queueFIFOQueueV2*
shapes
: *
shared_name *
capacity *
_output_shapes
: *
component_types
2*
	container 
Ё
?read_batch_features/file_name_queue/file_name_queue_EnqueueManyQueueEnqueueManyV2#read_batch_features/file_name_queue0read_batch_features/file_name_queue/limit_epochs*
Tcomponents
2*

timeout_ms€€€€€€€€€
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

SrcT0*
_output_shapes
: *

DstT0
n
)read_batch_features/file_name_queue/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   =
§
'read_batch_features/file_name_queue/mulMul(read_batch_features/file_name_queue/Cast)read_batch_features/file_name_queue/mul/y*
T0*
_output_shapes
: 
і
<read_batch_features/file_name_queue/fraction_of_32_full/tagsConst*H
value?B= B7read_batch_features/file_name_queue/fraction_of_32_full*
dtype0*
_output_shapes
: 
–
7read_batch_features/file_name_queue/fraction_of_32_fullScalarSummary<read_batch_features/file_name_queue/fraction_of_32_full/tags'read_batch_features/file_name_queue/mul*
_output_shapes
: *
T0
Х
)read_batch_features/read/TFRecordReaderV2TFRecordReaderV2*
shared_name *
compression_typeGZIP*
_output_shapes
: *
	container 
x
5read_batch_features/read/ReaderReadUpToV2/num_recordsConst*
_output_shapes
: *
dtype0	*
value
B	 Rи
ш
)read_batch_features/read/ReaderReadUpToV2ReaderReadUpToV2)read_batch_features/read/TFRecordReaderV2#read_batch_features/file_name_queue5read_batch_features/read/ReaderReadUpToV2/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
+read_batch_features/read/TFRecordReaderV2_1TFRecordReaderV2*
_output_shapes
: *
	container *
shared_name *
compression_typeGZIP
z
7read_batch_features/read/ReaderReadUpToV2_1/num_recordsConst*
value
B	 Rи*
_output_shapes
: *
dtype0	
ю
+read_batch_features/read/ReaderReadUpToV2_1ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_1#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_1/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
+read_batch_features/read/TFRecordReaderV2_2TFRecordReaderV2*
_output_shapes
: *
	container *
shared_name *
compression_typeGZIP
z
7read_batch_features/read/ReaderReadUpToV2_2/num_recordsConst*
value
B	 Rи*
dtype0	*
_output_shapes
: 
ю
+read_batch_features/read/ReaderReadUpToV2_2ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_2#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_2/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
+read_batch_features/read/TFRecordReaderV2_3TFRecordReaderV2*
_output_shapes
: *
	container *
shared_name *
compression_typeGZIP
z
7read_batch_features/read/ReaderReadUpToV2_3/num_recordsConst*
value
B	 Rи*
dtype0	*
_output_shapes
: 
ю
+read_batch_features/read/ReaderReadUpToV2_3ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_3#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_3/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
[
read_batch_features/ConstConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
І
read_batch_features/fifo_queueFIFOQueueV2*
shapes
: : *
shared_name *
capacity–*
	container *
_output_shapes
: *
component_types
2
В
read_batch_features/cond/SwitchSwitchread_batch_features/Constread_batch_features/Const*
T0
*
_output_shapes
: : 
q
!read_batch_features/cond/switch_tIdentity!read_batch_features/cond/Switch:1*
_output_shapes
: *
T0

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
6read_batch_features/cond/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue read_batch_features/cond/pred_id*
T0*
_output_shapes
: : *1
_class'
%#loc:@read_batch_features/fifo_queue
К
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1Switch)read_batch_features/read/ReaderReadUpToV2 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
М
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2Switch+read_batch_features/read/ReaderReadUpToV2:1 read_batch_features/cond/pred_id*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
©
/read_batch_features/cond/fifo_queue_EnqueueManyQueueEnqueueManyV28read_batch_features/cond/fifo_queue_EnqueueMany/Switch:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms€€€€€€€€€
г
+read_batch_features/cond/control_dependencyIdentity!read_batch_features/cond/switch_t0^read_batch_features/cond/fifo_queue_EnqueueMany*4
_class*
(&loc:@read_batch_features/cond/switch_t*
_output_shapes
: *
T0

I
read_batch_features/cond/NoOpNoOp"^read_batch_features/cond/switch_f
”
-read_batch_features/cond/control_dependency_1Identity!read_batch_features/cond/switch_f^read_batch_features/cond/NoOp*4
_class*
(&loc:@read_batch_features/cond/switch_f*
_output_shapes
: *
T0

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
8read_batch_features/cond_1/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_1/pred_id*
_output_shapes
: : *1
_class'
%#loc:@read_batch_features/fifo_queue*
T0
Т
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_1"read_batch_features/cond_1/pred_id*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1
Ф
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_1:1"read_batch_features/cond_1/pred_id*
T0*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_1/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms€€€€€€€€€
л
-read_batch_features/cond_1/control_dependencyIdentity#read_batch_features/cond_1/switch_t2^read_batch_features/cond_1/fifo_queue_EnqueueMany*
_output_shapes
: *6
_class,
*(loc:@read_batch_features/cond_1/switch_t*
T0

M
read_batch_features/cond_1/NoOpNoOp$^read_batch_features/cond_1/switch_f
џ
/read_batch_features/cond_1/control_dependency_1Identity#read_batch_features/cond_1/switch_f ^read_batch_features/cond_1/NoOp*
T0
*6
_class,
*(loc:@read_batch_features/cond_1/switch_f*
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
!read_batch_features/cond_2/SwitchSwitchread_batch_features/Constread_batch_features/Const*
_output_shapes
: : *
T0

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
%#loc:@read_batch_features/fifo_queue*
_output_shapes
: : *
T0
Т
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_2"read_batch_features/cond_2/pred_id*
T0*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_2:1"read_batch_features/cond_2/pred_id*
T0*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_2/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms€€€€€€€€€
л
-read_batch_features/cond_2/control_dependencyIdentity#read_batch_features/cond_2/switch_t2^read_batch_features/cond_2/fifo_queue_EnqueueMany*
T0
*6
_class,
*(loc:@read_batch_features/cond_2/switch_t*
_output_shapes
: 
M
read_batch_features/cond_2/NoOpNoOp$^read_batch_features/cond_2/switch_f
џ
/read_batch_features/cond_2/control_dependency_1Identity#read_batch_features/cond_2/switch_f ^read_batch_features/cond_2/NoOp*
T0
*
_output_shapes
: *6
_class,
*(loc:@read_batch_features/cond_2/switch_f
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
"read_batch_features/cond_3/pred_idIdentityread_batch_features/Const*
_output_shapes
: *
T0

№
8read_batch_features/cond_3/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_3/pred_id*
T0*
_output_shapes
: : *1
_class'
%#loc:@read_batch_features/fifo_queue
Т
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_3"read_batch_features/cond_3/pred_id*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Ф
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_3:1"read_batch_features/cond_3/pred_id*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0
±
1read_batch_features/cond_3/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms€€€€€€€€€
л
-read_batch_features/cond_3/control_dependencyIdentity#read_batch_features/cond_3/switch_t2^read_batch_features/cond_3/fifo_queue_EnqueueMany*
T0
*
_output_shapes
: *6
_class,
*(loc:@read_batch_features/cond_3/switch_t
M
read_batch_features/cond_3/NoOpNoOp$^read_batch_features/cond_3/switch_f
џ
/read_batch_features/cond_3/control_dependency_1Identity#read_batch_features/cond_3/switch_f ^read_batch_features/cond_3/NoOp*
T0
*
_output_shapes
: *6
_class,
*(loc:@read_batch_features/cond_3/switch_f
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

SrcT0*
_output_shapes
: *

DstT0
^
read_batch_features/mul/yConst*
valueB
 *o:*
_output_shapes
: *
dtype0
t
read_batch_features/mulMulread_batch_features/Castread_batch_features/mul/y*
_output_shapes
: *
T0
Ш
.read_batch_features/fraction_of_2000_full/tagsConst*:
value1B/ B)read_batch_features/fraction_of_2000_full*
_output_shapes
: *
dtype0
§
)read_batch_features/fraction_of_2000_fullScalarSummary.read_batch_features/fraction_of_2000_full/tagsread_batch_features/mul*
_output_shapes
: *
T0
X
read_batch_features/nConst*
dtype0*
_output_shapes
: *
value
B :и
 
read_batch_featuresQueueDequeueUpToV2read_batch_features/fifo_queueread_batch_features/n*

timeout_ms€€€€€€€€€*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
component_types
2
i
&read_batch_features/ParseExample/ConstConst*
valueB *
_output_shapes
: *
dtype0
k
(read_batch_features/ParseExample/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
k
(read_batch_features/ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB 
k
(read_batch_features/ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
k
(read_batch_features/ParseExample/Const_4Const*
valueB	 *
dtype0	*
_output_shapes
: 
k
(read_batch_features/ParseExample/Const_5Const*
_output_shapes
: *
dtype0	*
valueB	 
k
(read_batch_features/ParseExample/Const_6Const*
valueB	 *
_output_shapes
: *
dtype0	
k
(read_batch_features/ParseExample/Const_7Const*
dtype0	*
_output_shapes
: *
valueB	 
v
3read_batch_features/ParseExample/ParseExample/namesConst*
_output_shapes
: *
dtype0*
valueB 
А
:read_batch_features/ParseExample/ParseExample/dense_keys_0Const*
valueB Bkeyex*
dtype0*
_output_shapes
: 
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_1Const*
_output_shapes
: *
dtype0*
valueB Bnum1ex
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_2Const*
valueB Bnum2ex*
_output_shapes
: *
dtype0
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_3Const*
dtype0*
_output_shapes
: *
valueB Bnum3ex
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_4Const*
dtype0*
_output_shapes
: *
valueB Bstr1ex
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_5Const*
dtype0*
_output_shapes
: *
valueB Bstr2ex
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_6Const*
valueB Bstr3ex*
dtype0*
_output_shapes
: 
Г
:read_batch_features/ParseExample/ParseExample/dense_keys_7Const*
dtype0*
_output_shapes
: *
valueB Btargetex
≥	
-read_batch_features/ParseExample/ParseExampleParseExampleread_batch_features:13read_batch_features/ParseExample/ParseExample/names:read_batch_features/ParseExample/ParseExample/dense_keys_0:read_batch_features/ParseExample/ParseExample/dense_keys_1:read_batch_features/ParseExample/ParseExample/dense_keys_2:read_batch_features/ParseExample/ParseExample/dense_keys_3:read_batch_features/ParseExample/ParseExample/dense_keys_4:read_batch_features/ParseExample/ParseExample/dense_keys_5:read_batch_features/ParseExample/ParseExample/dense_keys_6:read_batch_features/ParseExample/ParseExample/dense_keys_7&read_batch_features/ParseExample/Const(read_batch_features/ParseExample/Const_1(read_batch_features/ParseExample/Const_2(read_batch_features/ParseExample/Const_3(read_batch_features/ParseExample/Const_4(read_batch_features/ParseExample/Const_5(read_batch_features/ParseExample/Const_6(read_batch_features/ParseExample/Const_7*
Nsparse *
Ndense*
Tdense

2				*"
dense_shapes
: : : : : : : : *
sparse_types
 *М
_output_shapesz
x:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ђ
 read_batch_features/fifo_queue_1FIFOQueueV2*
shapes
 *
_output_shapes
: * 
component_types
2					*
	container *
capacityd*
shared_name 
n
%read_batch_features/fifo_queue_1_SizeQueueSizeV2 read_batch_features/fifo_queue_1*
_output_shapes
: 
y
read_batch_features/Cast_1Cast%read_batch_features/fifo_queue_1_Size*
_output_shapes
: *

DstT0*

SrcT0
`
read_batch_features/mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
z
read_batch_features/mul_1Mulread_batch_features/Cast_1read_batch_features/mul_1/y*
_output_shapes
: *
T0
Д
dread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full/tagsConst*p
valuegBe B_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full*
dtype0*
_output_shapes
: 
Т
_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_fullScalarSummarydread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full/tagsread_batch_features/mul_1*
_output_shapes
: *
T0
∞
(read_batch_features/fifo_queue_1_enqueueQueueEnqueueV2 read_batch_features/fifo_queue_1-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:5/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7read_batch_features*
Tcomponents
2					*

timeout_ms€€€€€€€€€
≤
*read_batch_features/fifo_queue_1_enqueue_1QueueEnqueueV2 read_batch_features/fifo_queue_1-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:5/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7read_batch_features*
Tcomponents
2					*

timeout_ms€€€€€€€€€
w
&read_batch_features/fifo_queue_1_CloseQueueCloseV2 read_batch_features/fifo_queue_1*
cancel_pending_enqueues( 
y
(read_batch_features/fifo_queue_1_Close_1QueueCloseV2 read_batch_features/fifo_queue_1*
cancel_pending_enqueues(
є
(read_batch_features/fifo_queue_1_DequeueQueueDequeueV2 read_batch_features/fifo_queue_1*

timeout_ms€€€€€€€€€*Э
_output_shapesК
З:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
component_types
2					
Y
ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Т

ExpandDims
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:4ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:€€€€€€€€€
[
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
Ц
ExpandDims_1
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:1ExpandDims_1/dim*

Tdim0*'
_output_shapes
:€€€€€€€€€*
T0
[
ExpandDims_2/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Ф
ExpandDims_2
ExpandDims(read_batch_features/fifo_queue_1_DequeueExpandDims_2/dim*
T0*'
_output_shapes
:€€€€€€€€€*

Tdim0
[
ExpandDims_3/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ц
ExpandDims_3
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:2ExpandDims_3/dim*'
_output_shapes
:€€€€€€€€€*
T0*

Tdim0
[
ExpandDims_4/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ц
ExpandDims_4
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:5ExpandDims_4/dim*

Tdim0*
T0	*'
_output_shapes
:€€€€€€€€€
[
ExpandDims_5/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ц
ExpandDims_5
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:3ExpandDims_5/dim*

Tdim0*'
_output_shapes
:€€€€€€€€€*
T0
[
ExpandDims_6/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Ц
ExpandDims_6
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:6ExpandDims_6/dim*

Tdim0*'
_output_shapes
:€€€€€€€€€*
T0	
[
ExpandDims_7/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
Ц
ExpandDims_7
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:7ExpandDims_7/dim*

Tdim0*'
_output_shapes
:€€€€€€€€€*
T0	
Ѓ
ddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/ShapeShape
ExpandDims*
_output_shapes
:*
out_type0*
T0	
Е
cdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/CastCastddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Shape*
_output_shapes
:*

DstT0	*

SrcT0
≤
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast_1/xConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Ж
ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast_1Castgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast_1/x*
_output_shapes
: *

DstT0	*

SrcT0
Ш
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/NotEqualNotEqual
ExpandDimsednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast_1*
T0	*'
_output_shapes
:€€€€€€€€€
€
ddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/WhereWheregdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/NotEqual*'
_output_shapes
:€€€€€€€€€
њ
ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Reshape/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
І
fdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/ReshapeReshape
ExpandDimsldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€*
Tshape0
√
rdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stackConst*
valueB"       *
_output_shapes
:*
dtype0
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
б
ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_sliceStridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Whererdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stacktdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:€€€€€€€€€
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
«
vdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
«
vdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
н
ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1StridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Wheretdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stackvdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1vdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask*'
_output_shapes
:€€€€€€€€€*
end_mask*
T0	*
Index0*
shrink_axis_mask *
new_axis_mask 
П
fdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/unstackUnpackcdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
Р
ddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/stackPackhdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/unstack:1*
T0	*

axis *
N*
_output_shapes
:
с
bdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/MulMulndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1ddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/stack*
T0	*'
_output_shapes
:€€€€€€€€€
Њ
tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
О
bdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/SumSumbdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Multdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0	*
	keep_dims( *

Tidx0
й
bdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/AddAddldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slicebdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Sum*
T0	*#
_output_shapes
:€€€€€€€€€
Ч
ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/GatherGatherfdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Add*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0	*
Tindices0	
Т
Pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/mod/yConst*
value	B	 R	*
dtype0	*
_output_shapes
: 
Ѕ
Ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/modFloorModednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/GatherPdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/mod/y*
T0	*#
_output_shapes
:€€€€€€€€€
µ
kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ї
ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_sliceStridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Castkdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stackmdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stack_1mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stack_2*
Index0*
T0	*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
end_mask 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
є
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
є
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
√
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1StridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Castmdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stackodnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stack_1odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
T0	*
Index0*
_output_shapes
:*
shrink_axis_mask 
І
]dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
к
\dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/ProdProdgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1]dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
З
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat/values_1Pack\dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/Prod*
_output_shapes
:*
N*

axis *
T0	
•
cdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ў
^dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concatConcatV2ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slicegdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat/values_1cdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat/axis*
N*

Tidx0*
T0	*
_output_shapes
:
–
ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshapeSparseReshapeddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Wherecdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast^dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat*-
_output_shapes
:€€€€€€€€€:
ш
ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape/IdentityIdentityNdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/mod*#
_output_shapes
:€€€€€€€€€*
T0	
Е
adnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
valueB"	      
ш
`dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/meanConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 
ъ
bdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
valueB
 *Ђ™™>*
_output_shapes
: *
dtype0
Г
kdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaladnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/shape*
T0*
_output_shapes

:	*

seed *Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
dtype0*
seed2 
≥
_dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/mulMulkdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalbdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:	*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
T0
°
[dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normalAdd_dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/mul`dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/mean*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes

:	
Е
>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape
:	*
_output_shapes

:	*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
С
Ednn/input_from_feature_columns/str1ex_embedding/weights/part_0/AssignAssign>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0[dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal*
use_locking(*
validate_shape(*
T0*
_output_shapes

:	*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
Л
Cdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/readIdentity>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes

:	*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
T0
Є
ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
л
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SliceSlicegdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape:1ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice/beginmdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice/size*
Index0*
T0	*
_output_shapes
:
≤
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Б
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/ProdProdhdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slicehdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather/indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ю
idnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GatherGathergdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape:1qdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather/indices*
Tindices0*
validate_indices(*
Tparams0	*
_output_shapes
: 
Р
zdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/new_shapePackgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Prodidnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather*
_output_shapes
:*
N*

axis *
T0	
ь
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshapeSparseReshapeednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshapegdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape:1zdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:€€€€€€€€€:
£
ydnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/IdentityIdentityndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape/Identity*
T0	*#
_output_shapes
:€€€€€€€€€
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Ы
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GreaterEqualGreaterEqualydnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/Identityqdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
Л
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/WhereWhereodnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
√
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
Н
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/ReshapeReshapehdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape/shape*
Tshape0*#
_output_shapes
:€€€€€€€€€*
T0	
≥
kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_1Gatherpdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshapejdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*'
_output_shapes
:€€€€€€€€€
Є
kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_2Gatherydnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/Identityjdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:€€€€€€€€€
Р
kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/IdentityIdentityrdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
Њ
|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
’
Кdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
њ
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/IdentityКdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackМdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ї
{dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/CastCastДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice*
_output_shapes
: *

DstT0*

SrcT0	
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
љ
|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/rangeRangeВdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range/start{dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/CastВdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range/delta*#
_output_shapes
:€€€€€€€€€*

Tidx0
ј
}dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Cast_1Cast|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range*#
_output_shapes
:€€€€€€€€€*

DstT0	*

SrcT0
ё
Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
‘
Жdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_1Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackОdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
shrink_axis_mask*
T0	*
Index0*
end_mask*#
_output_shapes
:€€€€€€€€€*
ellipsis_mask *

begin_mask
я
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ListDiffListDiff}dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Cast_1Жdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
out_idx0*
T0	
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
«
Жdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/IdentityМdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackОdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0	*
Index0*
_output_shapes
: *
shrink_axis_mask
—
Еdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
“
Бdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDimsЖdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2Еdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
_output_shapes
: *
value	B
 Z
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
Ы
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDensednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ListDiffБdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ExpandDimsТdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesТdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*
Tindices0	*
validate_indices(*
T0
*#
_output_shapes
:€€€€€€€€€
÷
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
—
~dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ReshapeReshapednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ListDiffДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Reshape/shape*
T0	*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ѕ
Бdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLike~dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:€€€€€€€€€
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
ў
}dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concatConcatV2~dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ReshapeБdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/zeros_likeВdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat/axis*'
_output_shapes
:€€€€€€€€€*
N*
T0	*

Tidx0
ї
|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ShapeShapednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ListDiff*
out_type0*
_output_shapes
:*
T0	
≠
{dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/FillFill|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Shape|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:€€€€€€€€€
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
≈
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_1}dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concatДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*
T0	*'
_output_shapes
:€€€€€€€€€
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
њ
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_2{dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/FillДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:€€€€€€€€€*
N*
T0	*

Tidx0
∆
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorderdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_1dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_2kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Identity*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
T0	
Э
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/IdentityIdentitykdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Identity*
T0	*
_output_shapes
:
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ц
Иdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSliceДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseReorderОdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackРdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Рdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:€€€€€€€€€*
end_mask*
new_axis_mask *
ellipsis_mask *

begin_mask*
shrink_axis_mask*
T0	*
Index0
ѕ
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/CastCastИdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0	
б
Бdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/UniqueUniqueЖdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseReorder:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
out_idx0*
T0	
Т
Лdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherCdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/readБdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/Unique*'
_output_shapes
:€€€€€€€€€*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
Tparams0*
validate_indices(*
Tindices0	
в
zdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparseSparseSegmentMeanЛdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/embedding_lookupГdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/Unique:1dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:€€€€€€€€€
√
rdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
≤
ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_1ReshapeДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDenserdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_1/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0

Ґ
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/ShapeShapezdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse*
T0*
_output_shapes
:*
out_type0
ј
vdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
и
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_sliceStridedSlicehdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Shapevdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stackxdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stack_1xdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ђ
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
И
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/stackPackjdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/stack/0pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice*

axis *
_output_shapes
:*
T0*
N
Ф
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/TileTileldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_1hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/stack*

Tmultiples0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
®
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
т
bdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweightsSelectgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Tilemdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/zeros_likezdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
М
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/CastCastgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape:1*
_output_shapes
:*

DstT0*

SrcT0	
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
є
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
с
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1Slicegdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Castpdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1/beginodnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1/size*
_output_shapes
:*
Index0*
T0
М
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Shape_1Shapebdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights*
T0*
_output_shapes
:*
out_type0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
¬
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
ф
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2Slicejdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Shape_1pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2/beginodnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2/size*
Index0*
T0*
_output_shapes
:
∞
ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ч
idnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/concatConcatV2jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
Ж
ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_2Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweightsidnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/concat*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
∞
ddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/ShapeShapeExpandDims_4*
out_type0*
_output_shapes
:*
T0	
Е
cdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/CastCastddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Shape*

SrcT0*
_output_shapes
:*

DstT0	
≤
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
Ж
ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast_1Castgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0	
Ъ
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/NotEqualNotEqualExpandDims_4ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast_1*
T0	*'
_output_shapes
:€€€€€€€€€
€
ddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/WhereWheregdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/NotEqual*'
_output_shapes
:€€€€€€€€€
њ
ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
©
fdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/ReshapeReshapeExpandDims_4ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:€€€€€€€€€
√
rdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stackConst*
valueB"       *
_output_shapes
:*
dtype0
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
б
ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_sliceStridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Whererdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stacktdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
T0	*
Index0*
end_mask*#
_output_shapes
:€€€€€€€€€*
ellipsis_mask *

begin_mask
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stackConst*
valueB"        *
_output_shapes
:*
dtype0
«
vdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
«
vdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
н
ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1StridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Wheretdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stackvdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1vdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2*
end_mask*
ellipsis_mask *

begin_mask*
shrink_axis_mask *'
_output_shapes
:€€€€€€€€€*
new_axis_mask *
T0	*
Index0
П
fdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/unstackUnpackcdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast*	
num*
T0	*

axis *
_output_shapes
: : 
Р
ddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/stackPackhdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
с
bdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/MulMulndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1ddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/stack*
T0	*'
_output_shapes
:€€€€€€€€€
Њ
tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
О
bdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/SumSumbdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Multdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0	*#
_output_shapes
:€€€€€€€€€
й
bdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/AddAddldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slicebdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Sum*#
_output_shapes
:€€€€€€€€€*
T0	
Ч
ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/GatherGatherfdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Add*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0	*
Tindices0	
Т
Pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/mod/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
Ѕ
Ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/modFloorModednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/GatherPdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/mod/y*#
_output_shapes
:€€€€€€€€€*
T0	
µ
kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ї
ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_sliceStridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Castkdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stackmdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stack_1mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
_output_shapes
:*
end_mask *
Index0*
T0	*
shrink_axis_mask *
new_axis_mask 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
є
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
є
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
√
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1StridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Castmdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stackodnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stack_1odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0	
І
]dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
к
\dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/ProdProdgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1]dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
З
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat/values_1Pack\dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/Prod*
N*
T0	*
_output_shapes
:*

axis 
•
cdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ў
^dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concatConcatV2ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slicegdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat/values_1cdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat/axis*

Tidx0*
T0	*
N*
_output_shapes
:
–
ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshapeSparseReshapeddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Wherecdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast^dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat*-
_output_shapes
:€€€€€€€€€:
ш
ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape/IdentityIdentityNdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/mod*#
_output_shapes
:€€€€€€€€€*
T0	
Е
adnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
valueB"      *
_output_shapes
:*
dtype0
ш
`dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
valueB
 *    
ъ
bdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
valueB
 *уµ>
Г
kdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaladnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/shape*

seed *
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
seed2 *
dtype0*
_output_shapes

:
≥
_dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/mulMulkdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalbdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/stddev*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes

:
°
[dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normalAdd_dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/mul`dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/mean*
T0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
Е
>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
VariableV2*
shape
:*
_output_shapes

:*
shared_name *Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
dtype0*
	container 
С
Ednn/input_from_feature_columns/str2ex_embedding/weights/part_0/AssignAssign>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0[dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal*
_output_shapes

:*
validate_shape(*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
T0*
use_locking(
Л
Cdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/readIdentity>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes

:
Є
ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
л
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SliceSlicegdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape:1ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice/beginmdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice/size*
_output_shapes
:*
Index0*
T0	
≤
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Б
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/ProdProdhdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slicehdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather/indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ю
idnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GatherGathergdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape:1qdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather/indices*
_output_shapes
: *
validate_indices(*
Tparams0	*
Tindices0
Р
zdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/new_shapePackgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Prodidnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather*

axis *
_output_shapes
:*
T0	*
N
ь
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshapeSparseReshapeednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshapegdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape:1zdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:€€€€€€€€€:
£
ydnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/IdentityIdentityndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape/Identity*
T0	*#
_output_shapes
:€€€€€€€€€
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Ы
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GreaterEqualGreaterEqualydnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/Identityqdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
Л
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/WhereWhereodnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
√
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
Н
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/ReshapeReshapehdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:€€€€€€€€€
≥
kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_1Gatherpdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshapejdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape*'
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0	*
Tindices0	
Є
kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_2Gatherydnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/Identityjdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0	*
Tindices0	
Р
kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/IdentityIdentityrdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape:1*
_output_shapes
:*
T0	
Њ
|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
’
Кdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
њ
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/IdentityКdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackМdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*
_output_shapes
: *
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
T0	*
Index0
Ї
{dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/CastCastДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice*

SrcT0	*
_output_shapes
: *

DstT0
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range/startConst*
value	B : *
_output_shapes
: *
dtype0
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
љ
|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/rangeRangeВdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range/start{dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/CastВdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range/delta*#
_output_shapes
:€€€€€€€€€*

Tidx0
ј
}dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Cast_1Cast|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range*#
_output_shapes
:€€€€€€€€€*

DstT0	*

SrcT0
ё
Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
‘
Жdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_1Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackОdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*#
_output_shapes
:€€€€€€€€€*
end_mask*
new_axis_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask*
Index0*
T0	
я
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ListDiffListDiff}dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Cast_1Жdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
valueB: *
_output_shapes
:*
dtype0
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
«
Жdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/IdentityМdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackОdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0	*
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask 
—
Еdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
“
Бdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDimsЖdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2Еdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*
T0	*
_output_shapes
:*

Tdim0
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
_output_shapes
: *
dtype0
*
value	B
 Z
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
Ы
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDensednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ListDiffБdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ExpandDimsТdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesТdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
T0
*
Tindices0	
÷
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
—
~dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ReshapeReshapednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ListDiffДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Reshape/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0	
Ѕ
Бdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLike~dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Reshape*'
_output_shapes
:€€€€€€€€€*
T0	
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ў
}dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concatConcatV2~dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ReshapeБdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/zeros_likeВdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat/axis*
N*

Tidx0*
T0	*'
_output_shapes
:€€€€€€€€€
ї
|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ShapeShapednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ListDiff*
_output_shapes
:*
out_type0*
T0	
≠
{dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/FillFill|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Shape|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Const*#
_output_shapes
:€€€€€€€€€*
T0	
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
≈
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_1}dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concatДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_1/axis*

Tidx0*
T0	*
N*'
_output_shapes
:€€€€€€€€€
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
њ
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_2{dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/FillДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_2/axis*

Tidx0*
T0	*
N*#
_output_shapes
:€€€€€€€€€
∆
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorderdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_1dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_2kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Identity*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
T0	
Э
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/IdentityIdentitykdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Identity*
_output_shapes
:*
T0	
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ц
Иdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSliceДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseReorderОdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackРdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Рdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*#
_output_shapes
:€€€€€€€€€*
Index0*
T0	*
end_mask*
new_axis_mask *

begin_mask*
ellipsis_mask 
ѕ
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/CastCastИdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
б
Бdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/UniqueUniqueЖdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseReorder:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
out_idx0*
T0	
Т
Лdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherCdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/readБdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/Unique*
Tindices0	*
Tparams0*
validate_indices(*'
_output_shapes
:€€€€€€€€€*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
в
zdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparseSparseSegmentMeanЛdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/embedding_lookupГdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/Unique:1dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
√
rdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
≤
ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_1ReshapeДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDenserdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0

Ґ
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/ShapeShapezdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
ј
vdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
и
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_sliceStridedSlicehdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Shapevdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stackxdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stack_1xdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
ђ
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
И
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/stackPackjdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/stack/0pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice*
_output_shapes
:*
N*

axis *
T0
Ф
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/TileTileldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_1hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
*

Tmultiples0
®
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
т
bdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweightsSelectgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Tilemdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/zeros_likezdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
М
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/CastCastgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
є
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
с
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1Slicegdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Castpdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1/beginodnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1/size*
_output_shapes
:*
Index0*
T0
М
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Shape_1Shapebdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights*
out_type0*
_output_shapes
:*
T0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
¬
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
ф
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2Slicejdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Shape_1pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2/beginodnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2/size*
Index0*
T0*
_output_shapes
:
∞
ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ч
idnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/concatConcatV2jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
Ж
ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_2Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweightsidnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/concat*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
∞
ddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/ShapeShapeExpandDims_6*
_output_shapes
:*
out_type0*
T0	
Е
cdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/CastCastddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Shape*

SrcT0*
_output_shapes
:*

DstT0	
≤
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
Ж
ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast_1Castgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast_1/x*
_output_shapes
: *

DstT0	*

SrcT0
Ъ
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/NotEqualNotEqualExpandDims_6ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast_1*
T0	*'
_output_shapes
:€€€€€€€€€
€
ddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/WhereWheregdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/NotEqual*'
_output_shapes
:€€€€€€€€€
њ
ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
©
fdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/ReshapeReshapeExpandDims_6ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:€€€€€€€€€
√
rdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
б
ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_sliceStridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Whererdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stacktdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
«
vdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
«
vdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
н
ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1StridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Wheretdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stackvdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1vdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0	*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask 
П
fdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/unstackUnpackcdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast*	
num*
T0	*

axis *
_output_shapes
: : 
Р
ddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/stackPackhdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/unstack:1*
_output_shapes
:*
N*

axis *
T0	
с
bdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/MulMulndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1ddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/stack*'
_output_shapes
:€€€€€€€€€*
T0	
Њ
tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
О
bdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/SumSumbdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Multdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0	*#
_output_shapes
:€€€€€€€€€
й
bdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/AddAddldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slicebdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Sum*#
_output_shapes
:€€€€€€€€€*
T0	
Ч
ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/GatherGatherfdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Add*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:€€€€€€€€€
Т
Pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/mod/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
Ѕ
Ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/modFloorModednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/GatherPdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/mod/y*
T0	*#
_output_shapes
:€€€€€€€€€
µ
kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ї
ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_sliceStridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Castkdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stackmdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stack_1mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stack_2*
shrink_axis_mask *
_output_shapes
:*
Index0*
T0	*
end_mask *
new_axis_mask *

begin_mask*
ellipsis_mask 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
є
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
є
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
√
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1StridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Castmdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stackodnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stack_1odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0	*
_output_shapes
:*
shrink_axis_mask 
І
]dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/ConstConst*
valueB: *
_output_shapes
:*
dtype0
к
\dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/ProdProdgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1]dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
З
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat/values_1Pack\dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/Prod*
N*
T0	*
_output_shapes
:*

axis 
•
cdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ў
^dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concatConcatV2ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slicegdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat/values_1cdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat/axis*
_output_shapes
:*
T0	*

Tidx0*
N
–
ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshapeSparseReshapeddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Wherecdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast^dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat*-
_output_shapes
:€€€€€€€€€:
ш
ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape/IdentityIdentityNdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/mod*
T0	*#
_output_shapes
:€€€€€€€€€
Е
adnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
valueB"      *
dtype0*
_output_shapes
:
ш
`dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/meanConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
valueB
 *    *
_output_shapes
: *
dtype0
ъ
bdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
valueB
 *уµ>
Г
kdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaladnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
≥
_dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/mulMulkdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalbdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/stddev*
T0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
°
[dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normalAdd_dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/mul`dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/mean*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
_output_shapes

:
Е
>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
VariableV2*
shared_name *Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:
С
Ednn/input_from_feature_columns/str3ex_embedding/weights/part_0/AssignAssign>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0[dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal*
use_locking(*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
validate_shape(*
_output_shapes

:
Л
Cdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/readIdentity>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
T0
Є
ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
л
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SliceSlicegdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape:1ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice/beginmdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice/size*
Index0*
T0	*
_output_shapes
:
≤
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Б
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/ProdProdhdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slicehdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Ю
idnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GatherGathergdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape:1qdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather/indices*
_output_shapes
: *
validate_indices(*
Tparams0	*
Tindices0
Р
zdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/new_shapePackgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Prodidnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather*

axis *
_output_shapes
:*
T0	*
N
ь
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshapeSparseReshapeednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshapegdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape:1zdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:€€€€€€€€€:
£
ydnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/IdentityIdentityndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape/Identity*
T0	*#
_output_shapes
:€€€€€€€€€
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
Ы
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GreaterEqualGreaterEqualydnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/Identityqdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
Л
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/WhereWhereodnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
√
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
Н
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/ReshapeReshapehdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape/shape*
Tshape0*#
_output_shapes
:€€€€€€€€€*
T0	
≥
kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_1Gatherpdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshapejdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*'
_output_shapes
:€€€€€€€€€
Є
kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_2Gatherydnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/Identityjdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:€€€€€€€€€
Р
kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/IdentityIdentityrdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
Њ
|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ConstConst*
value	B	 R *
_output_shapes
: *
dtype0	
’
Кdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
њ
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/IdentityКdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackМdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*
end_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask*
_output_shapes
: *
new_axis_mask *
T0	*
Index0
Ї
{dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/CastCastДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice*

SrcT0	*
_output_shapes
: *

DstT0
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
љ
|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/rangeRangeВdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range/start{dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/CastВdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:€€€€€€€€€
ј
}dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Cast_1Cast|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range*

SrcT0*#
_output_shapes
:€€€€€€€€€*

DstT0	
ё
Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
‘
Жdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_1Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackОdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *#
_output_shapes
:€€€€€€€€€*
end_mask*
Index0*
T0	*
shrink_axis_mask*
new_axis_mask 
я
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ListDiffListDiff}dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Cast_1Жdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
valueB: *
_output_shapes
:*
dtype0
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
«
Жdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/IdentityМdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackОdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0	*
Index0*
end_mask *
new_axis_mask *
ellipsis_mask *

begin_mask 
—
Еdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
“
Бdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDimsЖdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2Еdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*
T0	*
_output_shapes
:*

Tdim0
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
_output_shapes
: *
dtype0
*
value	B
 Z
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
Ы
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDensednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ListDiffБdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ExpandDimsТdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesТdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
T0
*
Tindices0	
÷
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
—
~dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ReshapeReshapednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ListDiffДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Reshape/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0	
Ѕ
Бdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLike~dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:€€€€€€€€€
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ў
}dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concatConcatV2~dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ReshapeБdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/zeros_likeВdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat/axis*
N*

Tidx0*
T0	*'
_output_shapes
:€€€€€€€€€
ї
|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ShapeShapednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ListDiff*
T0	*
_output_shapes
:*
out_type0
≠
{dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/FillFill|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Shape|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Const*#
_output_shapes
:€€€€€€€€€*
T0	
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
≈
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_1}dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concatДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:€€€€€€€€€*
T0	*

Tidx0*
N
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
њ
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_2{dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/FillДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:€€€€€€€€€*
N*
T0	*

Tidx0
∆
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorderdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_1dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_2kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Identity*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
T0	
Э
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/IdentityIdentitykdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Identity*
T0	*
_output_shapes
:
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ц
Иdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSliceДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseReorderОdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackРdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Рdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*
ellipsis_mask *

begin_mask*#
_output_shapes
:€€€€€€€€€*
end_mask*
T0	*
Index0*
shrink_axis_mask*
new_axis_mask 
ѕ
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/CastCastИdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
б
Бdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/UniqueUniqueЖdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseReorder:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
out_idx0*
T0	
Т
Лdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherCdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/readБdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/Unique*
Tindices0	*
Tparams0*
validate_indices(*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*'
_output_shapes
:€€€€€€€€€
в
zdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparseSparseSegmentMeanЛdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/embedding_lookupГdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/Unique:1dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
√
rdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
≤
ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_1ReshapeДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDenserdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_1/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0

Ґ
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/ShapeShapezdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
ј
vdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
и
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_sliceStridedSlicehdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Shapevdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stackxdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stack_1xdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
ђ
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
И
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/stackPackjdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/stack/0pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
Ф
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/TileTileldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_1hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
*

Tmultiples0
®
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
т
bdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweightsSelectgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Tilemdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/zeros_likezdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
М
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/CastCastgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
є
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
с
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1Slicegdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Castpdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1/beginodnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1/size*
_output_shapes
:*
Index0*
T0
М
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Shape_1Shapebdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights*
out_type0*
_output_shapes
:*
T0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
¬
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
ф
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2Slicejdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Shape_1pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2/beginodnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2/size*
Index0*
T0*
_output_shapes
:
∞
ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ч
idnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/concatConcatV2jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
Ж
ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_2Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweightsidnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/concat*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
З
Ednn/input_from_feature_columns/input_from_feature_columns/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
ќ
@dnn/input_from_feature_columns/input_from_feature_columns/concatConcatV2ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_2ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_2ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_2ExpandDims_1ExpandDims_3ExpandDims_5Ednn/input_from_feature_columns/input_from_feature_columns/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:€€€€€€€€€	
«
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"	   
   *
_output_shapes
:*
dtype0
є
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *№њ
є
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *№?
°
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
seed2 *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*

seed *
_output_shapes

:	
*
T0
Ю
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
∞
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*
_output_shapes

:	
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0
Ґ
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*
T0*
_output_shapes

:	
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
…
 dnn/hiddenlayer_0/weights/part_0
VariableV2*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes

:	
*
shape
:	
*
dtype0*
shared_name *
	container 
Ч
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0*
_output_shapes

:	
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
±
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes

:	
*
T0
≤
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueB
*    *
dtype0*
_output_shapes
:

њ
dnn/hiddenlayer_0/biases/part_0
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
	container *
shape:
*
dtype0*
_output_shapes
:

Ж
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0
™
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0
u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
_output_shapes

:	
*
T0
„
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
_output_shapes
:
*
T0
°
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*'
_output_shapes
:€€€€€€€€€
*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*'
_output_shapes
:€€€€€€€€€
*
T0
W
zero_fraction/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    
И
zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*'
_output_shapes
:€€€€€€€€€
*
T0
p
zero_fraction/CastCastzero_fraction/Equal*

SrcT0
*'
_output_shapes
:€€€€€€€€€
*

DstT0
d
zero_fraction/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
Б
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ш
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values*
dtype0*
_output_shapes
: 
Я
)dnn/hiddenlayer_0_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0_fraction_of_zero_values/tagszero_fraction/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0_activation/tagConst*
dtype0*
_output_shapes
: *-
value$B" Bdnn/hiddenlayer_0_activation
Щ
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
_output_shapes
: *
T0
«
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"
      
є
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *:Ќњ
є
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *:Ќ?*
_output_shapes
: *
dtype0
°
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:
*
dtype0*
seed2 *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*

seed 
Ю
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: *
T0
∞
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0
Ґ
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes

:
*
T0
…
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
Ч
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
_output_shapes

:
*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
use_locking(
±
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes

:
*
T0
≤
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueB*    *
dtype0*
_output_shapes
:
њ
dnn/hiddenlayer_1/biases/part_0
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
Ж
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
validate_shape(*
_output_shapes
:
™
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:

ї
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *'
_output_shapes
:€€€€€€€€€*
transpose_a( *
T0
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:
°
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Y
zero_fraction_1/zeroConst*
_output_shapes
: *
dtype0*
valueB
 *    
М
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*'
_output_shapes
:€€€€€€€€€*
T0
t
zero_fraction_1/CastCastzero_fraction_1/Equal*

SrcT0
*'
_output_shapes
:€€€€€€€€€*

DstT0
f
zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
З
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ш
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values
°
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
_output_shapes
: *
T0
}
 dnn/hiddenlayer_1_activation/tagConst*-
value$B" Bdnn/hiddenlayer_1_activation*
dtype0*
_output_shapes
: 
Щ
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
_output_shapes
: *
T0
«
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"      *
dtype0*
_output_shapes
:
є
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *тк-њ*
_output_shapes
: *
dtype0
є
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *тк-?*
_output_shapes
: *
dtype0
°
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
seed2 *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*

seed *
_output_shapes

:*
T0
Ю
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
∞
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes

:
Ґ
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*
T0*
_output_shapes

:*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
…
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape
:*
_output_shapes

:*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
Ч
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
_output_shapes

:*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
use_locking(
±
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes

:
≤
1dnn/hiddenlayer_2/biases/part_0/Initializer/ConstConst*
dtype0*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB*    
њ
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
:*
shape:*
shared_name 
Ж
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/Const*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
™
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
:*
T0
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
T0*
_output_shapes

:
ї
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *'
_output_shapes
:€€€€€€€€€*
transpose_a( *
T0
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
:
°
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*
data_formatNHWC*
T0*'
_output_shapes
:€€€€€€€€€
y
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Y
zero_fraction_2/zeroConst*
_output_shapes
: *
dtype0*
valueB
 *    
М
zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reluzero_fraction_2/zero*
T0*'
_output_shapes
:€€€€€€€€€
t
zero_fraction_2/CastCastzero_fraction_2/Equal*'
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

f
zero_fraction_2/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
З
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Ш
.dnn/hiddenlayer_2_fraction_of_zero_values/tagsConst*:
value1B/ B)dnn/hiddenlayer_2_fraction_of_zero_values*
dtype0*
_output_shapes
: 
°
)dnn/hiddenlayer_2_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2_fraction_of_zero_values/tagszero_fraction_2/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_2_activation/tagConst*
dtype0*
_output_shapes
: *-
value$B" Bdnn/hiddenlayer_2_activation
Щ
dnn/hiddenlayer_2_activationHistogramSummary dnn/hiddenlayer_2_activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
є
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      *
dtype0*
_output_shapes
:
Ђ
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *„≥]њ
Ђ
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *„≥]?
М
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes

:*
T0*
dtype0*
seed2 *

seed 
В
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *,
_class"
 loc:@dnn/logits/weights/part_0
Ф
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes

:*
T0
Ж
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*
T0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes

:
ї
dnn/logits/weights/part_0
VariableV2*
	container *
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes

:*
shape
:*
shared_name 
ы
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0
Ь
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes

:*
T0
§
*dnn/logits/biases/part_0/Initializer/ConstConst*
_output_shapes
:*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    
±
dnn/logits/biases/part_0
VariableV2*
	container *
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
:*
shape:*
shared_name 
к
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0
Х
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*
T0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
_output_shapes

:*
T0
≠
dnn/logits/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/logits/weights*
transpose_b( *'
_output_shapes
:€€€€€€€€€*
transpose_a( *
T0
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
М
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*'
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
T0
Y
zero_fraction_3/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
zero_fraction_3/EqualEqualdnn/logits/BiasAddzero_fraction_3/zero*
T0*'
_output_shapes
:€€€€€€€€€
t
zero_fraction_3/CastCastzero_fraction_3/Equal*

SrcT0
*'
_output_shapes
:€€€€€€€€€*

DstT0
f
zero_fraction_3/ConstConst*
valueB"       *
_output_shapes
:*
dtype0
З
zero_fraction_3/MeanMeanzero_fraction_3/Castzero_fraction_3/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
К
'dnn/logits_fraction_of_zero_values/tagsConst*3
value*B( B"dnn/logits_fraction_of_zero_values*
dtype0*
_output_shapes
: 
У
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_3/Mean*
_output_shapes
: *
T0
o
dnn/logits_activation/tagConst*
dtype0*
_output_shapes
: *&
valueB Bdnn/logits_activation
y
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
_output_shapes
: *
T0
j
predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
_
predictions/classes/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
К
predictions/classesArgMaxdnn/logits/BiasAddpredictions/classes/dimension*#
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
О
0training_loss/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_7*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
.training_loss/softmax_cross_entropy_loss/ShapeShape0training_loss/softmax_cross_entropy_loss/Squeeze*
_output_shapes
:*
out_type0*
T0	
е
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
Tlabels0	
]
training_loss/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Т
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
e
 training_loss/ScalarSummary/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bloss
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
_output_shapes
: *
T0
У
,metrics/remove_squeezable_dimensions/SqueezeSqueezeExpandDims_7*
squeeze_dims

€€€€€€€€€*#
_output_shapes
:€€€€€€€€€*
T0	
З
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:€€€€€€€€€
c
metrics/ToFloatCastmetrics/Equal*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
[
metrics/accuracy/zerosConst*
valueB
 *    *
_output_shapes
: *
dtype0
z
metrics/accuracy/total
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
ћ
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
_output_shapes
: *
validate_shape(*)
_class
loc:@metrics/accuracy/total*
T0*
use_locking(
Л
metrics/accuracy/total/readIdentitymetrics/accuracy/total*
T0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
]
metrics/accuracy/zeros_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
z
metrics/accuracy/count
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
ќ
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
_output_shapes
: *
validate_shape(*)
_class
loc:@metrics/accuracy/count*
T0*
use_locking(
Л
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: *
T0
_
metrics/accuracy/SizeSizemetrics/ToFloat*
T0*
out_type0*
_output_shapes
: 
i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*
_output_shapes
: *

DstT0*

SrcT0
`
metrics/accuracy/ConstConst*
valueB: *
dtype0*
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
loc:@metrics/accuracy/total*
_output_shapes
: *
T0*
use_locking( 
Љ
metrics/accuracy/AssignAdd_1	AssignAddmetrics/accuracy/countmetrics/accuracy/ToFloat_1*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: *
T0*
use_locking( 
_
metrics/accuracy/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
}
metrics/accuracy/GreaterGreatermetrics/accuracy/count/readmetrics/accuracy/Greater/y*
T0*
_output_shapes
: 
~
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
_output_shapes
: *
T0
]
metrics/accuracy/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
П
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
_output_shapes
: *
T0
a
metrics/accuracy/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
В
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
А
metrics/accuracy/truediv_1RealDivmetrics/accuracy/AssignAddmetrics/accuracy/AssignAdd_1*
_output_shapes
: *
T0
a
metrics/accuracy/update_op/eConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
N
metrics/RankConst*
value	B :*
dtype0*
_output_shapes
: 
U
metrics/LessEqual/yConst*
value	B :*
dtype0*
_output_shapes
: 
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
_output_shapes
: *
T0
Т
metrics/Assert/ConstConst*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
dtype0*
_output_shapes
: 
Ъ
metrics/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*

T
2*
	summarize
А
metrics/Reshape/shapeConst^metrics/Assert/Assert*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
{
metrics/ReshapeReshapeExpandDims_7metrics/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:€€€€€€€€€
]
metrics/one_hot/on_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
^
metrics/one_hot/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
metrics/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
«
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
T0*
TI0	*
axis€€€€€€€€€*'
_output_shapes
:€€€€€€€€€
f
metrics/CastCastmetrics/one_hot*'
_output_shapes
:€€€€€€€€€*

DstT0
*

SrcT0
j
metrics/auc/Reshape/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
Ф
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
l
metrics/auc/Reshape_1/shapeConst*
valueB"   €€€€*
_output_shapes
:*
dtype0
Л
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€*
Tshape0
d
metrics/auc/ShapeShapemetrics/auc/Reshape*
_output_shapes
:*
out_type0*
T0
i
metrics/auc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!metrics/auc/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
k
!metrics/auc/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
µ
metrics/auc/strided_sliceStridedSlicemetrics/auc/Shapemetrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*
ellipsis_mask *

begin_mask *
end_mask 
А
metrics/auc/ConstConst*
_output_shapes	
:»*
dtype0*є
valueѓBђ»"†Хњ÷≥ѕ©§;ѕ©$<Јюv<ѕ©§<C‘Ќ<Јюц<Х=ѕ©$=	?9=C‘M=}ib=Јюv=ш…Е=ХР=2_Ъ=ѕ©§=lфЃ=	?є=¶Й√=C‘Ќ=аЎ=}iв=ім=Јюц=™§ >ш…>Gп
>Х>д9>2_>БД>ѕ©$>ѕ)>lф.>ї4>	?9>Wd>>¶ЙC>фЃH>C‘M>СщR>аX>.D]>}ib>ЋОg>іl>hўq>Јюv>$|>™§А>Q7Г>ш…Е>†\И>GпК>оБН>ХР><ІТ>д9Х>ЛћЧ>2_Ъ>ўсЬ>БДЯ>(Ґ>ѕ©§>v<І>ѕ©>≈aђ>lфЃ>З±>їі>bђґ>	?є>∞—ї>WdЊ>€цј>¶Й√>M∆>фЃ»>ЬAЋ>C‘Ќ>кf–>Сщ“>9М’>аЎ>З±Џ>.DЁ>÷÷я>}iв>$ьд>ЋОз>r!к>ім>ЅFп>hўс>lф>Јюц>^Сщ>$ь>ђґю>™§ ?эн?Q7?•А?ш…?L?†\?у•	?Gп
?Ъ8?оБ?BЋ?Х?й]?<І?Рр?д9?7Г?Лћ?я?2_?Ж®?ўс?-;?БД?‘Ќ ?("?{`#?ѕ©$?#у%?v<'? Е(?ѕ)?q+?≈a,?Ђ-?lф.?ј=0?З1?g–2?ї4?c5?bђ6?µх7?	?9?]И:?∞—;?=?Wd>?Ђ≠??€ц@?R@B?¶ЙC?ъ“D?MF?°eG?фЃH?HшI?ЬAK?пКL?C‘M?ЧO?кfP?>∞Q?СщR?еBT?9МU?М’V?аX?3hY?З±Z?џъ[?.D]?ВН^?÷÷_?) a?}ib?–≤c?$ьd?xEf?ЋОg?Ўh?r!j?∆jk?іl?mэm?ЅFo?Рp?hўq?Љ"s?lt?cµu?Јюv?
Hx?^Сy?≤Џz?$|?Ym}?ђґ~? А?
d
metrics/auc/ExpandDims/dimConst*
valueB:*
_output_shapes
:*
dtype0
Й
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*
T0*
_output_shapes
:	»*

Tdim0
U
metrics/auc/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
Г
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*

axis *
_output_shapes
:*
T0*
N
И
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:»€€€€€€€€€
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
_output_shapes
: *
T0
]
metrics/auc/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Ѓ
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
_output_shapes
:*
T0
У
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*'
_output_shapes
:€€€€€€€€€*
T0
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
_output_shapes
:*
valueB"»      
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
dtype0*
_output_shapes
:*
valueB"»      
Ф
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*(
_output_shapes
:»€€€€€€€€€*
T0
*

Tmultiples0
d
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2*(
_output_shapes
:»€€€€€€€€€
`
metrics/auc/zerosConst*
_output_shapes	
:»*
dtype0*
valueB»*    
И
metrics/auc/true_positives
VariableV2*
shape:»*
shared_name *
dtype0*
_output_shapes	
:»*
	container 
Ў
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes	
:»*
T0*
validate_shape(*
use_locking(
Ь
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*
T0*
_output_shapes	
:»*-
_class#
!loc:@metrics/auc/true_positives
w
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
w
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

SrcT0
*(
_output_shapes
:»€€€€€€€€€*

DstT0
c
!metrics/auc/Sum/reduction_indicesConst*
value	B :*
dtype0*
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
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*
use_locking( *
T0*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes	
:»
b
metrics/auc/zeros_1Const*
valueB»*    *
_output_shapes	
:»*
dtype0
Й
metrics/auc/false_negatives
VariableV2*
shared_name *
dtype0*
shape:»*
_output_shapes	
:»*
	container 
Ё
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes	
:»*
T0*
validate_shape(*
use_locking(
Я
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*
_output_shapes	
:»*.
_class$
" loc:@metrics/auc/false_negatives*
T0
|
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

SrcT0
*(
_output_shapes
:»€€€€€€€€€*

DstT0
e
#metrics/auc/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
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
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*
use_locking( *
T0*
_output_shapes	
:»*.
_class$
" loc:@metrics/auc/false_negatives
b
metrics/auc/zeros_2Const*
valueB»*    *
_output_shapes	
:»*
dtype0
И
metrics/auc/true_negatives
VariableV2*
shape:»*
shared_name *
dtype0*
_output_shapes	
:»*
	container 
Џ
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
_output_shapes	
:»*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
use_locking(
Ь
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*
_output_shapes	
:»*-
_class#
!loc:@metrics/auc/true_negatives*
T0
В
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*(
_output_shapes
:»€€€€€€€€€*

DstT0*

SrcT0

e
#metrics/auc/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ч
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*
_output_shapes	
:»*
T0*
	keep_dims( *

Tidx0
ї
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*
use_locking( *
T0*-
_class#
!loc:@metrics/auc/true_negatives*
_output_shapes	
:»
b
metrics/auc/zeros_3Const*
_output_shapes	
:»*
dtype0*
valueB»*    
Й
metrics/auc/false_positives
VariableV2*
shared_name *
dtype0*
shape:»*
_output_shapes	
:»*
	container 
Ё
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:»*.
_class$
" loc:@metrics/auc/false_positives
Я
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*
T0*.
_class$
" loc:@metrics/auc/false_positives*
_output_shapes	
:»

metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

SrcT0
*(
_output_shapes
:»€€€€€€€€€*

DstT0
e
#metrics/auc/Sum_3/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
Ч
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:»
љ
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*
use_locking( *
T0*
_output_shapes	
:»*.
_class$
" loc:@metrics/auc/false_positives
V
metrics/auc/add/yConst*
valueB
 *љ7Ж5*
_output_shapes
: *
dtype0
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
_output_shapes	
:»*
T0
Б
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:»
X
metrics/auc/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
_output_shapes	
:»*
T0
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
_output_shapes	
:»*
T0
Б
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:»
X
metrics/auc/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:»
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
_output_shapes	
:»*
T0
k
!metrics/auc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
n
#metrics/auc/strided_slice_1/stack_1Const*
valueB:«*
dtype0*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
¬
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
shrink_axis_mask *
T0*
Index0*
end_mask *
_output_shapes	
:«*

begin_mask*
ellipsis_mask 
k
!metrics/auc/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
¬
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes	
:«
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:«
k
!metrics/auc/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
valueB:«*
_output_shapes
:*
dtype0
m
#metrics/auc/strided_slice_3/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ј
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes	
:«
k
!metrics/auc/strided_slice_4/stackConst*
valueB:*
_output_shapes
:*
dtype0
m
#metrics/auc/strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ј
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
shrink_axis_mask *
T0*
Index0*
end_mask*
_output_shapes	
:«*

begin_mask *
ellipsis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
_output_shapes	
:«*
T0
Z
metrics/auc/truediv/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
T0*
_output_shapes	
:«
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
_output_shapes	
:«*
T0
]
metrics/auc/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
|
metrics/auc/valueSummetrics/auc/Mulmetrics/auc/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
metrics/auc/add_6/yConst*
valueB
 *љ7Ж5*
dtype0*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
_output_shapes	
:»*
T0
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:»
X
metrics/auc/add_8/yConst*
dtype0*
_output_shapes
: *
valueB
 *љ7Ж5
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:»
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
_output_shapes	
:»*
T0
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
_output_shapes	
:»*
T0
Y
metrics/auc/add_10/yConst*
valueB
 *љ7Ж5*
dtype0*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
_output_shapes	
:»*
T0
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
_output_shapes	
:»*
T0
k
!metrics/auc/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:«
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
¬
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes	
:«*
shrink_axis_mask 
k
!metrics/auc/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
m
#metrics/auc/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
¬
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes	
:«*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
end_mask
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
_output_shapes	
:«*
T0
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
_output_shapes
:*
valueB: 
n
#metrics/auc/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:«
m
#metrics/auc/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¬
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
T0*
Index0*
_output_shapes	
:«*
shrink_axis_mask 
k
!metrics/auc/strided_slice_8/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#metrics/auc/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
m
#metrics/auc/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¬
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0*
_output_shapes	
:«*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
_output_shapes	
:«*
T0
\
metrics/auc/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
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
metrics/auc/Const_2Const*
valueB: *
_output_shapes
:*
dtype0
В
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
И
*metrics/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_7*
squeeze_dims
*#
_output_shapes
:€€€€€€€€€*
T0	
Т
(metrics/softmax_cross_entropy_loss/ShapeShape*metrics/softmax_cross_entropy_loss/Squeeze*
T0	*
_output_shapes
:*
out_type0
ў
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
Tlabels0	*
T0
a
metrics/eval_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ф
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
W
metrics/mean/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    
v
metrics/mean/total
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
Љ
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
use_locking(*
T0*%
_class
loc:@metrics/mean/total*
validate_shape(*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*
T0*
_output_shapes
: *%
_class
loc:@metrics/mean/total
Y
metrics/mean/zeros_1Const*
valueB
 *    *
_output_shapes
: *
dtype0
v
metrics/mean/count
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
Њ
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
_output_shapes
: *
validate_shape(*%
_class
loc:@metrics/mean/count*
T0*
use_locking(

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
_output_shapes
: *
T0
S
metrics/mean/SizeConst*
dtype0*
_output_shapes
: *
value	B :
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

SrcT0*
_output_shapes
: *

DstT0
U
metrics/mean/ConstConst*
valueB *
_output_shapes
: *
dtype0
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
loc:@metrics/mean/total*
_output_shapes
: *
T0*
use_locking( 
ђ
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
_output_shapes
: *
T0*
use_locking( 
[
metrics/mean/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
_output_shapes
: *
T0
Y
metrics/mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
_output_shapes
: *
T0
]
metrics/mean/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
_output_shapes
: *
T0
]
metrics/mean/update_op/eConst*
valueB
 *    *
_output_shapes
: *
dtype0
Л
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
_output_shapes
: *
T0
`

group_depsNoOp^metrics/mean/update_op^metrics/auc/update_op^metrics/accuracy/update_op
\
eval_step/initial_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
m
	eval_step
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
¶
eval_step/AssignAssign	eval_stepeval_step/initial_value*
use_locking(*
T0*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
T0*
_output_shapes
: *
_class
loc:@eval_step
T
AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@eval_step
Ј
initNoOp^global_step/AssignF^dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/AssignF^dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/AssignF^dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign(^dnn/hiddenlayer_2/weights/part_0/Assign'^dnn/hiddenlayer_2/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
З
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
З
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitialized>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
З
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
Ћ
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
…
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0
Ћ
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
…
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: *
dtype0
Ћ
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: *
dtype0
…
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
Њ
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializeddnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
dtype0*
_output_shapes
: 
Љ
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializeddnn/logits/biases/part_0*
_output_shapes
: *
dtype0*+
_class!
loc:@dnn/logits/biases/part_0
ъ
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized7read_batch_features/file_name_queue/limit_epochs/epochs*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
dtype0	*
_output_shapes
: 
Є
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedmetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: *
dtype0
Є
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedmetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
dtype0*
_output_shapes
: 
ј
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedmetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes
: *
dtype0
¬
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedmetrics/auc/false_negatives*
dtype0*
_output_shapes
: *.
_class$
" loc:@metrics/auc/false_negatives
ј
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedmetrics/auc/true_negatives*
_output_shapes
: *
dtype0*-
_class#
!loc:@metrics/auc/true_negatives
¬
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedmetrics/auc/false_positives*
dtype0*
_output_shapes
: *.
_class$
" loc:@metrics/auc/false_positives
∞
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedmetrics/mean/total*%
_class
loc:@metrics/mean/total*
_output_shapes
: *
dtype0
∞
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitializedmetrics/mean/count*
dtype0*
_output_shapes
: *%
_class
loc:@metrics/mean/count
Ю
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized	eval_step*
dtype0*
_output_shapes
: *
_class
loc:@eval_step
ј

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_21*
N*
T0
*
_output_shapes
:*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
«
$report_uninitialized_variables/ConstConst*о
valueдBбBglobal_stepB>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0B>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0B>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B7read_batch_features/file_name_queue/limit_epochs/epochsBmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/countB	eval_step*
_output_shapes
:*
dtype0
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ў
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask *
Index0*
T0*
end_mask *
_output_shapes
:*

begin_mask*
ellipsis_mask 
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
х
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
б
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
end_mask*
T0*
Index0*
shrink_axis_mask *
new_axis_mask 
ѓ
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ђ
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
Ћ
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:*
T0
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
џ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
Tshape0*
_output_shapes
:
Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:€€€€€€€€€
ґ
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:€€€€€€€€€*
T0	
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:€€€€€€€€€
g
$report_uninitialized_resources/ConstConst*
_output_shapes
: *
dtype0*
valueB 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
°
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Й
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
Й
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitialized>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
Й
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
_output_shapes
: *
dtype0
Ќ
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: *
dtype0
Ћ
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
dtype0*
_output_shapes
: 
Ќ
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: *
dtype0
Ћ
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0
Ќ
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: *
dtype0
Ћ
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
ј
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitializeddnn/logits/weights/part_0*
dtype0*
_output_shapes
: *,
_class"
 loc:@dnn/logits/weights/part_0
Њ
9report_uninitialized_variables_1/IsVariableInitialized_11IsVariableInitializeddnn/logits/biases/part_0*
dtype0*
_output_shapes
: *+
_class!
loc:@dnn/logits/biases/part_0
†
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_109report_uninitialized_variables_1/IsVariableInitialized_11*
N*
T0
*
_output_shapes
:*

axis 
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
ї
&report_uninitialized_variables_1/ConstConst*
_output_shapes
:*
dtype0*а
value÷B”Bglobal_stepB>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0B>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0B>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
г
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
Index0*
T0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
end_mask 
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
ы
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
_output_shapes
: *
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask *
Index0*
T0
≥
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
_output_shapes
:*
N*

axis *
T0
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
≥
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
—
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
T0*
Tshape0*
_output_shapes
:
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
б
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
_output_shapes
:*
Tshape0*
T0

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
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0*
Tindices0	
м
init_2NoOp?^read_batch_features/file_name_queue/limit_epochs/epochs/Assign^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign^eval_step/Assign

init_all_tablesNoOp
/
group_deps_2NoOp^init_2^init_all_tables
ї
Merge/MergeSummaryMergeSummary7read_batch_features/file_name_queue/fraction_of_32_full)read_batch_features/fraction_of_2000_full_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full)dnn/hiddenlayer_0_fraction_of_zero_valuesdnn/hiddenlayer_0_activation)dnn/hiddenlayer_1_fraction_of_zero_valuesdnn/hiddenlayer_1_activation)dnn/hiddenlayer_2_fraction_of_zero_valuesdnn/hiddenlayer_2_activation"dnn/logits_fraction_of_zero_valuesdnn/logits_activationtraining_loss/ScalarSummary*
_output_shapes
: *
N
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_369eff4c4b72430cb3790865237e6ed1/part*
dtype0*
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
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
а
save/SaveV2/tensor_namesConst*
_output_shapes
:*
dtype0*У
valueЙBЖBdnn/hiddenlayer_0/biasesBdnn/hiddenlayer_0/weightsBdnn/hiddenlayer_1/biasesBdnn/hiddenlayer_1/weightsBdnn/hiddenlayer_2/biasesBdnn/hiddenlayer_2/weightsB7dnn/input_from_feature_columns/str1ex_embedding/weightsB7dnn/input_from_feature_columns/str2ex_embedding/weightsB7dnn/input_from_feature_columns/str3ex_embedding/weightsBdnn/logits/biasesBdnn/logits/weightsBglobal_step
е
save/SaveV2/shape_and_slicesConst*Ф
valueКBЗB10 0,10B9 10 0,9:0,10B8 0,8B10 8 0,10:0,8B5 0,5B8 5 0,8:0,5B9 2 0,9:0,2B8 2 0,8:0,2B8 2 0,8:0,2B3 0,3B5 3 0,5:0,3B *
dtype0*
_output_shapes
:
Б
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read%dnn/hiddenlayer_0/weights/part_0/read$dnn/hiddenlayer_1/biases/part_0/read%dnn/hiddenlayer_1/weights/part_0/read$dnn/hiddenlayer_2/biases/part_0/read%dnn/hiddenlayer_2/weights/part_0/readCdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/readCdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/readCdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/readdnn/logits/biases/part_0/readdnn/logits/weights/part_0/readglobal_step*
dtypes
2	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
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
|
save/RestoreV2/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_0/biases*
_output_shapes
:*
dtype0
o
save/RestoreV2/shape_and_slicesConst*
valueBB10 0,10*
dtype0*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
»
save/AssignAssigndnn/hiddenlayer_0/biases/part_0save/RestoreV2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0

save/RestoreV2_1/tensor_namesConst*.
value%B#Bdnn/hiddenlayer_0/weights*
_output_shapes
:*
dtype0
w
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
dtype0*"
valueBB9 10 0,9:0,10
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
“
save/Assign_1Assign dnn/hiddenlayer_0/weights/part_0save/RestoreV2_1*
use_locking(*
validate_shape(*
T0*
_output_shapes

:	
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
~
save/RestoreV2_2/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bdnn/hiddenlayer_1/biases
o
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueBB8 0,8
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
ћ
save/Assign_2Assigndnn/hiddenlayer_1/biases/part_0save/RestoreV2_2*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*.
value%B#Bdnn/hiddenlayer_1/weights
w
!save/RestoreV2_3/shape_and_slicesConst*"
valueBB10 8 0,10:0,8*
_output_shapes
:*
dtype0
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
“
save/Assign_3Assign dnn/hiddenlayer_1/weights/part_0save/RestoreV2_3*
use_locking(*
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
validate_shape(*
_output_shapes

:

~
save/RestoreV2_4/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_2/biases*
dtype0*
_output_shapes
:
o
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueBB5 0,5
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
ћ
save/Assign_4Assigndnn/hiddenlayer_2/biases/part_0save/RestoreV2_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0

save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*
dtype0*.
value%B#Bdnn/hiddenlayer_2/weights
u
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
_output_shapes
:* 
valueBB8 5 0,8:0,5
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
“
save/Assign_5Assign dnn/hiddenlayer_2/weights/part_0save/RestoreV2_5*
use_locking(*
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
validate_shape(*
_output_shapes

:
Э
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*
dtype0*L
valueCBAB7dnn/input_from_feature_columns/str1ex_embedding/weights
u
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
_output_shapes
:* 
valueBB9 2 0,9:0,2
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
О
save/Assign_6Assign>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0save/RestoreV2_6*
use_locking(*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
validate_shape(*
_output_shapes

:	
Э
save/RestoreV2_7/tensor_namesConst*L
valueCBAB7dnn/input_from_feature_columns/str2ex_embedding/weights*
dtype0*
_output_shapes
:
u
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0* 
valueBB8 2 0,8:0,2
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
О
save/Assign_7Assign>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0save/RestoreV2_7*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes

:*
T0*
validate_shape(*
use_locking(
Э
save/RestoreV2_8/tensor_namesConst*L
valueCBAB7dnn/input_from_feature_columns/str3ex_embedding/weights*
_output_shapes
:*
dtype0
u
!save/RestoreV2_8/shape_and_slicesConst* 
valueBB8 2 0,8:0,2*
_output_shapes
:*
dtype0
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
О
save/Assign_8Assign>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0save/RestoreV2_8*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
w
save/RestoreV2_9/tensor_namesConst*&
valueBBdnn/logits/biases*
dtype0*
_output_shapes
:
o
!save/RestoreV2_9/shape_and_slicesConst*
valueBB3 0,3*
dtype0*
_output_shapes
:
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Њ
save/Assign_9Assigndnn/logits/biases/part_0save/RestoreV2_9*
_output_shapes
:*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
use_locking(
y
save/RestoreV2_10/tensor_namesConst*'
valueBBdnn/logits/weights*
dtype0*
_output_shapes
:
v
"save/RestoreV2_10/shape_and_slicesConst* 
valueBB5 3 0,5:0,3*
dtype0*
_output_shapes
:
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
∆
save/Assign_10Assigndnn/logits/weights/part_0save/RestoreV2_10*
use_locking(*
T0*,
_class"
 loc:@dnn/logits/weights/part_0*
validate_shape(*
_output_shapes

:
r
save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBglobal_step
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Щ
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2	*
_output_shapes
:
Ґ
save/Assign_11Assignglobal_stepsave/RestoreV2_11*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
Џ
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11
-
save/restore_allNoOp^save/restore_shard"[Хєг/     Рw…≥	/n.K°9÷AJҐК
Ш;ф:
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
S
HistogramSummary
tag
values"T
summary"
Ttype0:
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

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
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
D
NotEqual
x"T
y"T
z
"
Ttype:
2	
Р
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
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
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
A
Relu
features"T
activations"T"
Ttype:
2		
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
z
SparseSegmentMean	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2"
Tidxtype0:
2	
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

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
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
Ttype*1.0.12v1.0.0-65-g4763edf-dirtyРф

global_step/Initializer/ConstConst*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
value	B	 R 
П
global_step
VariableV2*
	container *
dtype0	*
_class
loc:@global_step*
shared_name *
_output_shapes
: *
shape: 
≤
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_output_shapes
: *
_class
loc:@global_step
¶
)read_batch_features/file_name_queue/inputConst*I
value@B>B4../tfpreout/features_eval-00000-of-00001.tfrecord.gz*
dtype0*
_output_shapes
:
j
(read_batch_features/file_name_queue/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
o
-read_batch_features/file_name_queue/Greater/yConst*
value	B : *
_output_shapes
: *
dtype0
∞
+read_batch_features/file_name_queue/GreaterGreater(read_batch_features/file_name_queue/Size-read_batch_features/file_name_queue/Greater/y*
_output_shapes
: *
T0
І
0read_batch_features/file_name_queue/Assert/ConstConst*G
value>B< B6string_input_producer requires a non-null input tensor*
dtype0*
_output_shapes
: 
ѓ
8read_batch_features/file_name_queue/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor
њ
1read_batch_features/file_name_queue/Assert/AssertAssert+read_batch_features/file_name_queue/Greater8read_batch_features/file_name_queue/Assert/Assert/data_0*

T
2*
	summarize
Љ
,read_batch_features/file_name_queue/IdentityIdentity)read_batch_features/file_name_queue/input2^read_batch_features/file_name_queue/Assert/Assert*
_output_shapes
:*
T0
x
6read_batch_features/file_name_queue/limit_epochs/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Ы
7read_batch_features/file_name_queue/limit_epochs/epochs
VariableV2*
_output_shapes
: *
	container *
dtype0	*
shared_name *
shape: 
ѕ
>read_batch_features/file_name_queue/limit_epochs/epochs/AssignAssign7read_batch_features/file_name_queue/limit_epochs/epochs6read_batch_features/file_name_queue/limit_epochs/Const*
_output_shapes
: *
validate_shape(*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
T0	*
use_locking(
о
<read_batch_features/file_name_queue/limit_epochs/epochs/readIdentity7read_batch_features/file_name_queue/limit_epochs/epochs*
T0	*
_output_shapes
: *J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs
ъ
:read_batch_features/file_name_queue/limit_epochs/CountUpTo	CountUpTo7read_batch_features/file_name_queue/limit_epochs/epochs*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
_output_shapes
: *
limit*
T0	
ћ
0read_batch_features/file_name_queue/limit_epochsIdentity,read_batch_features/file_name_queue/Identity;^read_batch_features/file_name_queue/limit_epochs/CountUpTo*
_output_shapes
:*
T0
®
#read_batch_features/file_name_queueFIFOQueueV2*
shapes
: *
	container *
shared_name *
_output_shapes
: *
component_types
2*
capacity 
Ё
?read_batch_features/file_name_queue/file_name_queue_EnqueueManyQueueEnqueueManyV2#read_batch_features/file_name_queue0read_batch_features/file_name_queue/limit_epochs*
Tcomponents
2*

timeout_ms€€€€€€€€€
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
(read_batch_features/file_name_queue/CastCast8read_batch_features/file_name_queue/file_name_queue_Size*
_output_shapes
: *

DstT0*

SrcT0
n
)read_batch_features/file_name_queue/mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *   =
§
'read_batch_features/file_name_queue/mulMul(read_batch_features/file_name_queue/Cast)read_batch_features/file_name_queue/mul/y*
_output_shapes
: *
T0
і
<read_batch_features/file_name_queue/fraction_of_32_full/tagsConst*H
value?B= B7read_batch_features/file_name_queue/fraction_of_32_full*
dtype0*
_output_shapes
: 
–
7read_batch_features/file_name_queue/fraction_of_32_fullScalarSummary<read_batch_features/file_name_queue/fraction_of_32_full/tags'read_batch_features/file_name_queue/mul*
_output_shapes
: *
T0
Х
)read_batch_features/read/TFRecordReaderV2TFRecordReaderV2*
_output_shapes
: *
	container *
shared_name *
compression_typeGZIP
x
5read_batch_features/read/ReaderReadUpToV2/num_recordsConst*
dtype0	*
_output_shapes
: *
value
B	 Rи
ш
)read_batch_features/read/ReaderReadUpToV2ReaderReadUpToV2)read_batch_features/read/TFRecordReaderV2#read_batch_features/file_name_queue5read_batch_features/read/ReaderReadUpToV2/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
+read_batch_features/read/TFRecordReaderV2_1TFRecordReaderV2*
shared_name *
compression_typeGZIP*
_output_shapes
: *
	container 
z
7read_batch_features/read/ReaderReadUpToV2_1/num_recordsConst*
value
B	 Rи*
_output_shapes
: *
dtype0	
ю
+read_batch_features/read/ReaderReadUpToV2_1ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_1#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_1/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
+read_batch_features/read/TFRecordReaderV2_2TFRecordReaderV2*
shared_name *
compression_typeGZIP*
_output_shapes
: *
	container 
z
7read_batch_features/read/ReaderReadUpToV2_2/num_recordsConst*
_output_shapes
: *
dtype0	*
value
B	 Rи
ю
+read_batch_features/read/ReaderReadUpToV2_2ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_2#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_2/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ч
+read_batch_features/read/TFRecordReaderV2_3TFRecordReaderV2*
shared_name *
compression_typeGZIP*
_output_shapes
: *
	container 
z
7read_batch_features/read/ReaderReadUpToV2_3/num_recordsConst*
value
B	 Rи*
dtype0	*
_output_shapes
: 
ю
+read_batch_features/read/ReaderReadUpToV2_3ReaderReadUpToV2+read_batch_features/read/TFRecordReaderV2_3#read_batch_features/file_name_queue7read_batch_features/read/ReaderReadUpToV2_3/num_records*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
[
read_batch_features/ConstConst*
dtype0
*
_output_shapes
: *
value	B
 Z
І
read_batch_features/fifo_queueFIFOQueueV2*
shapes
: : *
	container *
shared_name *
_output_shapes
: *
component_types
2*
capacity–
В
read_batch_features/cond/SwitchSwitchread_batch_features/Constread_batch_features/Const*
_output_shapes
: : *
T0

q
!read_batch_features/cond/switch_tIdentity!read_batch_features/cond/Switch:1*
_output_shapes
: *
T0

o
!read_batch_features/cond/switch_fIdentityread_batch_features/cond/Switch*
_output_shapes
: *
T0

h
 read_batch_features/cond/pred_idIdentityread_batch_features/Const*
T0
*
_output_shapes
: 
Ў
6read_batch_features/cond/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue read_batch_features/cond/pred_id*
T0*
_output_shapes
: : *1
_class'
%#loc:@read_batch_features/fifo_queue
К
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1Switch)read_batch_features/read/ReaderReadUpToV2 read_batch_features/cond/pred_id*
T0*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
М
8read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2Switch+read_batch_features/read/ReaderReadUpToV2:1 read_batch_features/cond/pred_id*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*<
_class2
0.loc:@read_batch_features/read/ReaderReadUpToV2
©
/read_batch_features/cond/fifo_queue_EnqueueManyQueueEnqueueManyV28read_batch_features/cond/fifo_queue_EnqueueMany/Switch:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms€€€€€€€€€
г
+read_batch_features/cond/control_dependencyIdentity!read_batch_features/cond/switch_t0^read_batch_features/cond/fifo_queue_EnqueueMany*
T0
*4
_class*
(&loc:@read_batch_features/cond/switch_t*
_output_shapes
: 
I
read_batch_features/cond/NoOpNoOp"^read_batch_features/cond/switch_f
”
-read_batch_features/cond/control_dependency_1Identity!read_batch_features/cond/switch_f^read_batch_features/cond/NoOp*
T0
*
_output_shapes
: *4
_class*
(&loc:@read_batch_features/cond/switch_f
ѓ
read_batch_features/cond/MergeMerge-read_batch_features/cond/control_dependency_1+read_batch_features/cond/control_dependency*
T0
*
N*
_output_shapes
: : 
Д
!read_batch_features/cond_1/SwitchSwitchread_batch_features/Constread_batch_features/Const*
_output_shapes
: : *
T0

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
"read_batch_features/cond_1/pred_idIdentityread_batch_features/Const*
_output_shapes
: *
T0

№
8read_batch_features/cond_1/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_1/pred_id*
_output_shapes
: : *1
_class'
%#loc:@read_batch_features/fifo_queue*
T0
Т
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_1"read_batch_features/cond_1/pred_id*
T0*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_1:1"read_batch_features/cond_1/pred_id*
T0*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_1/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms€€€€€€€€€
л
-read_batch_features/cond_1/control_dependencyIdentity#read_batch_features/cond_1/switch_t2^read_batch_features/cond_1/fifo_queue_EnqueueMany*
_output_shapes
: *6
_class,
*(loc:@read_batch_features/cond_1/switch_t*
T0

M
read_batch_features/cond_1/NoOpNoOp$^read_batch_features/cond_1/switch_f
џ
/read_batch_features/cond_1/control_dependency_1Identity#read_batch_features/cond_1/switch_f ^read_batch_features/cond_1/NoOp*
T0
*6
_class,
*(loc:@read_batch_features/cond_1/switch_f*
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
#read_batch_features/cond_2/switch_tIdentity#read_batch_features/cond_2/Switch:1*
_output_shapes
: *
T0

s
#read_batch_features/cond_2/switch_fIdentity!read_batch_features/cond_2/Switch*
T0
*
_output_shapes
: 
j
"read_batch_features/cond_2/pred_idIdentityread_batch_features/Const*
_output_shapes
: *
T0

№
8read_batch_features/cond_2/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_2/pred_id*
T0*1
_class'
%#loc:@read_batch_features/fifo_queue*
_output_shapes
: : 
Т
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_2"read_batch_features/cond_2/pred_id*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*
T0
Ф
:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_2:1"read_batch_features/cond_2/pred_id*
T0*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
1read_batch_features/cond_2/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms€€€€€€€€€
л
-read_batch_features/cond_2/control_dependencyIdentity#read_batch_features/cond_2/switch_t2^read_batch_features/cond_2/fifo_queue_EnqueueMany*
_output_shapes
: *6
_class,
*(loc:@read_batch_features/cond_2/switch_t*
T0

M
read_batch_features/cond_2/NoOpNoOp$^read_batch_features/cond_2/switch_f
џ
/read_batch_features/cond_2/control_dependency_1Identity#read_batch_features/cond_2/switch_f ^read_batch_features/cond_2/NoOp*6
_class,
*(loc:@read_batch_features/cond_2/switch_f*
_output_shapes
: *
T0

µ
 read_batch_features/cond_2/MergeMerge/read_batch_features/cond_2/control_dependency_1-read_batch_features/cond_2/control_dependency*
_output_shapes
: : *
N*
T0

Д
!read_batch_features/cond_3/SwitchSwitchread_batch_features/Constread_batch_features/Const*
_output_shapes
: : *
T0

u
#read_batch_features/cond_3/switch_tIdentity#read_batch_features/cond_3/Switch:1*
T0
*
_output_shapes
: 
s
#read_batch_features/cond_3/switch_fIdentity!read_batch_features/cond_3/Switch*
_output_shapes
: *
T0

j
"read_batch_features/cond_3/pred_idIdentityread_batch_features/Const*
_output_shapes
: *
T0

№
8read_batch_features/cond_3/fifo_queue_EnqueueMany/SwitchSwitchread_batch_features/fifo_queue"read_batch_features/cond_3/pred_id*1
_class'
%#loc:@read_batch_features/fifo_queue*
_output_shapes
: : *
T0
Т
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1Switch+read_batch_features/read/ReaderReadUpToV2_3"read_batch_features/cond_3/pred_id*
T0*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ф
:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2Switch-read_batch_features/read/ReaderReadUpToV2_3:1"read_batch_features/cond_3/pred_id*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*>
_class4
20loc:@read_batch_features/read/ReaderReadUpToV2_3*
T0
±
1read_batch_features/cond_3/fifo_queue_EnqueueManyQueueEnqueueManyV2:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms€€€€€€€€€
л
-read_batch_features/cond_3/control_dependencyIdentity#read_batch_features/cond_3/switch_t2^read_batch_features/cond_3/fifo_queue_EnqueueMany*
T0
*6
_class,
*(loc:@read_batch_features/cond_3/switch_t*
_output_shapes
: 
M
read_batch_features/cond_3/NoOpNoOp$^read_batch_features/cond_3/switch_f
џ
/read_batch_features/cond_3/control_dependency_1Identity#read_batch_features/cond_3/switch_f ^read_batch_features/cond_3/NoOp*6
_class,
*(loc:@read_batch_features/cond_3/switch_f*
_output_shapes
: *
T0

µ
 read_batch_features/cond_3/MergeMerge/read_batch_features/cond_3/control_dependency_1-read_batch_features/cond_3/control_dependency*
T0
*
N*
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
read_batch_features/CastCast#read_batch_features/fifo_queue_Size*
_output_shapes
: *

DstT0*

SrcT0
^
read_batch_features/mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:
t
read_batch_features/mulMulread_batch_features/Castread_batch_features/mul/y*
T0*
_output_shapes
: 
Ш
.read_batch_features/fraction_of_2000_full/tagsConst*
dtype0*
_output_shapes
: *:
value1B/ B)read_batch_features/fraction_of_2000_full
§
)read_batch_features/fraction_of_2000_fullScalarSummary.read_batch_features/fraction_of_2000_full/tagsread_batch_features/mul*
T0*
_output_shapes
: 
X
read_batch_features/nConst*
dtype0*
_output_shapes
: *
value
B :и
 
read_batch_featuresQueueDequeueUpToV2read_batch_features/fifo_queueread_batch_features/n*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
component_types
2*

timeout_ms€€€€€€€€€
i
&read_batch_features/ParseExample/ConstConst*
valueB *
dtype0*
_output_shapes
: 
k
(read_batch_features/ParseExample/Const_1Const*
valueB *
_output_shapes
: *
dtype0
k
(read_batch_features/ParseExample/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
k
(read_batch_features/ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
k
(read_batch_features/ParseExample/Const_4Const*
valueB	 *
dtype0	*
_output_shapes
: 
k
(read_batch_features/ParseExample/Const_5Const*
_output_shapes
: *
dtype0	*
valueB	 
k
(read_batch_features/ParseExample/Const_6Const*
valueB	 *
dtype0	*
_output_shapes
: 
k
(read_batch_features/ParseExample/Const_7Const*
valueB	 *
dtype0	*
_output_shapes
: 
v
3read_batch_features/ParseExample/ParseExample/namesConst*
_output_shapes
: *
dtype0*
valueB 
А
:read_batch_features/ParseExample/ParseExample/dense_keys_0Const*
valueB Bkeyex*
_output_shapes
: *
dtype0
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_1Const*
dtype0*
_output_shapes
: *
valueB Bnum1ex
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_2Const*
valueB Bnum2ex*
dtype0*
_output_shapes
: 
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_3Const*
dtype0*
_output_shapes
: *
valueB Bnum3ex
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_4Const*
valueB Bstr1ex*
_output_shapes
: *
dtype0
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_5Const*
dtype0*
_output_shapes
: *
valueB Bstr2ex
Б
:read_batch_features/ParseExample/ParseExample/dense_keys_6Const*
dtype0*
_output_shapes
: *
valueB Bstr3ex
Г
:read_batch_features/ParseExample/ParseExample/dense_keys_7Const*
valueB Btargetex*
_output_shapes
: *
dtype0
≥	
-read_batch_features/ParseExample/ParseExampleParseExampleread_batch_features:13read_batch_features/ParseExample/ParseExample/names:read_batch_features/ParseExample/ParseExample/dense_keys_0:read_batch_features/ParseExample/ParseExample/dense_keys_1:read_batch_features/ParseExample/ParseExample/dense_keys_2:read_batch_features/ParseExample/ParseExample/dense_keys_3:read_batch_features/ParseExample/ParseExample/dense_keys_4:read_batch_features/ParseExample/ParseExample/dense_keys_5:read_batch_features/ParseExample/ParseExample/dense_keys_6:read_batch_features/ParseExample/ParseExample/dense_keys_7&read_batch_features/ParseExample/Const(read_batch_features/ParseExample/Const_1(read_batch_features/ParseExample/Const_2(read_batch_features/ParseExample/Const_3(read_batch_features/ParseExample/Const_4(read_batch_features/ParseExample/Const_5(read_batch_features/ParseExample/Const_6(read_batch_features/ParseExample/Const_7*
Nsparse *М
_output_shapesz
x:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
Ndense*
Tdense

2				*"
dense_shapes
: : : : : : : : *
sparse_types
 
Ђ
 read_batch_features/fifo_queue_1FIFOQueueV2*
shapes
 *
	container *
_output_shapes
: * 
component_types
2					*
capacityd*
shared_name 
n
%read_batch_features/fifo_queue_1_SizeQueueSizeV2 read_batch_features/fifo_queue_1*
_output_shapes
: 
y
read_batch_features/Cast_1Cast%read_batch_features/fifo_queue_1_Size*

SrcT0*
_output_shapes
: *

DstT0
`
read_batch_features/mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
z
read_batch_features/mul_1Mulread_batch_features/Cast_1read_batch_features/mul_1/y*
T0*
_output_shapes
: 
Д
dread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full/tagsConst*p
valuegBe B_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full*
dtype0*
_output_shapes
: 
Т
_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_fullScalarSummarydread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full/tagsread_batch_features/mul_1*
_output_shapes
: *
T0
∞
(read_batch_features/fifo_queue_1_enqueueQueueEnqueueV2 read_batch_features/fifo_queue_1-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:5/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7read_batch_features*
Tcomponents
2					*

timeout_ms€€€€€€€€€
≤
*read_batch_features/fifo_queue_1_enqueue_1QueueEnqueueV2 read_batch_features/fifo_queue_1-read_batch_features/ParseExample/ParseExample/read_batch_features/ParseExample/ParseExample:1/read_batch_features/ParseExample/ParseExample:2/read_batch_features/ParseExample/ParseExample:3/read_batch_features/ParseExample/ParseExample:4/read_batch_features/ParseExample/ParseExample:5/read_batch_features/ParseExample/ParseExample:6/read_batch_features/ParseExample/ParseExample:7read_batch_features*
Tcomponents
2					*

timeout_ms€€€€€€€€€
w
&read_batch_features/fifo_queue_1_CloseQueueCloseV2 read_batch_features/fifo_queue_1*
cancel_pending_enqueues( 
y
(read_batch_features/fifo_queue_1_Close_1QueueCloseV2 read_batch_features/fifo_queue_1*
cancel_pending_enqueues(
є
(read_batch_features/fifo_queue_1_DequeueQueueDequeueV2 read_batch_features/fifo_queue_1*

timeout_ms€€€€€€€€€*Э
_output_shapesК
З:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
component_types
2					
Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
Т

ExpandDims
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:4ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0	*

Tdim0
[
ExpandDims_1/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Ц
ExpandDims_1
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:1ExpandDims_1/dim*
T0*'
_output_shapes
:€€€€€€€€€*

Tdim0
[
ExpandDims_2/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Ф
ExpandDims_2
ExpandDims(read_batch_features/fifo_queue_1_DequeueExpandDims_2/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
[
ExpandDims_3/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Ц
ExpandDims_3
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:2ExpandDims_3/dim*
T0*'
_output_shapes
:€€€€€€€€€*

Tdim0
[
ExpandDims_4/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Ц
ExpandDims_4
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:5ExpandDims_4/dim*

Tdim0*
T0	*'
_output_shapes
:€€€€€€€€€
[
ExpandDims_5/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
Ц
ExpandDims_5
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:3ExpandDims_5/dim*
T0*'
_output_shapes
:€€€€€€€€€*

Tdim0
[
ExpandDims_6/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
Ц
ExpandDims_6
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:6ExpandDims_6/dim*
T0	*'
_output_shapes
:€€€€€€€€€*

Tdim0
[
ExpandDims_7/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
Ц
ExpandDims_7
ExpandDims*read_batch_features/fifo_queue_1_Dequeue:7ExpandDims_7/dim*
T0	*'
_output_shapes
:€€€€€€€€€*

Tdim0
Ѓ
ddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/ShapeShape
ExpandDims*
T0	*
out_type0*
_output_shapes
:
Е
cdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/CastCastddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Shape*

SrcT0*
_output_shapes
:*

DstT0	
≤
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast_1/xConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ж
ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast_1Castgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast_1/x*
_output_shapes
: *

DstT0	*

SrcT0
Ш
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/NotEqualNotEqual
ExpandDimsednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast_1*'
_output_shapes
:€€€€€€€€€*
T0	
€
ddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/WhereWheregdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/NotEqual*'
_output_shapes
:€€€€€€€€€
њ
ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
І
fdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/ReshapeReshape
ExpandDimsldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:€€€€€€€€€
√
rdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stackConst*
valueB"       *
dtype0*
_output_shapes
:
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
б
ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_sliceStridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Whererdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stacktdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
«
vdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
«
vdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
н
ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1StridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Wheretdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stackvdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1vdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask*'
_output_shapes
:€€€€€€€€€*
end_mask*
T0	*
Index0*
shrink_axis_mask *
new_axis_mask 
П
fdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/unstackUnpackcdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast*

axis *
_output_shapes
: : *	
num*
T0	
Р
ddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/stackPackhdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
с
bdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/MulMulndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slice_1ddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/stack*
T0	*'
_output_shapes
:€€€€€€€€€
Њ
tdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
О
bdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/SumSumbdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Multdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0	*
	keep_dims( *

Tidx0
й
bdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/AddAddldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/strided_slicebdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Sum*
T0	*#
_output_shapes
:€€€€€€€€€
Ч
ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/GatherGatherfdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Add*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:€€€€€€€€€
Т
Pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/mod/yConst*
_output_shapes
: *
dtype0	*
value	B	 R	
Ѕ
Ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/modFloorModednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/GatherPdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/mod/y*#
_output_shapes
:€€€€€€€€€*
T0	
µ
kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ї
ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_sliceStridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Castkdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stackmdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stack_1mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
_output_shapes
:*
end_mask *
Index0*
T0	*
shrink_axis_mask *
new_axis_mask 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
є
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
є
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
√
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1StridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Castmdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stackodnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stack_1odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
:*
end_mask*
T0	*
Index0*
shrink_axis_mask *
new_axis_mask 
І
]dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
к
\dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/ProdProdgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slice_1]dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
З
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat/values_1Pack\dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/Prod*
N*
T0	*
_output_shapes
:*

axis 
•
cdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ў
^dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concatConcatV2ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/strided_slicegdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat/values_1cdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat/axis*

Tidx0*
T0	*
N*
_output_shapes
:
–
ednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshapeSparseReshapeddnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Wherecdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/DenseToSparseTensor/Cast^dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/concat*-
_output_shapes
:€€€€€€€€€:
ш
ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape/IdentityIdentityNdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/mod*
T0	*#
_output_shapes
:€€€€€€€€€
Е
adnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
valueB"	      *
_output_shapes
:*
dtype0
ш
`dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/meanConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 
ъ
bdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
valueB
 *Ђ™™>
Г
kdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaladnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes

:	
≥
_dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/mulMulkdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalbdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/stddev*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes

:	
°
[dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normalAdd_dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/mul`dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal/mean*
T0*
_output_shapes

:	*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
Е
>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
VariableV2*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes

:	*
shape
:	*
dtype0*
shared_name *
	container 
С
Ednn/input_from_feature_columns/str1ex_embedding/weights/part_0/AssignAssign>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0[dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/Initializer/truncated_normal*
use_locking(*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
validate_shape(*
_output_shapes

:	
Л
Cdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/readIdentity>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes

:	*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
T0
Є
ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
л
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SliceSlicegdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape:1ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice/beginmdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice/size*
_output_shapes
:*
Index0*
T0	
≤
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Б
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/ProdProdhdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slicehdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather/indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ю
idnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GatherGathergdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape:1qdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather/indices*
_output_shapes
: *
validate_indices(*
Tparams0	*
Tindices0
Р
zdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/new_shapePackgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Prodidnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather*
T0	*

axis *
N*
_output_shapes
:
ь
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshapeSparseReshapeednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshapegdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape:1zdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:€€€€€€€€€:
£
ydnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/IdentityIdentityndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape/Identity*#
_output_shapes
:€€€€€€€€€*
T0	
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Ы
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GreaterEqualGreaterEqualydnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/Identityqdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
Л
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/WhereWhereodnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
√
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
Н
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/ReshapeReshapehdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€*
Tshape0
≥
kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_1Gatherpdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshapejdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape*'
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0	*
Tindices0	
Є
kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_2Gatherydnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape/Identityjdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0	*
Tindices0	
Р
kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/IdentityIdentityrdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
Њ
|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
’
Кdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
њ
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/IdentityКdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackМdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*
_output_shapes
: *
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
Index0*
T0	
Ї
{dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/CastCastДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice*
_output_shapes
: *

DstT0*

SrcT0	
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
љ
|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/rangeRangeВdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range/start{dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/CastВdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range/delta*#
_output_shapes
:€€€€€€€€€*

Tidx0
ј
}dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Cast_1Cast|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/range*

SrcT0*#
_output_shapes
:€€€€€€€€€*

DstT0	
ё
Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
‘
Жdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_1Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackОdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*#
_output_shapes
:€€€€€€€€€*
end_mask*
new_axis_mask *
ellipsis_mask *

begin_mask*
shrink_axis_mask*
T0	*
Index0
я
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ListDiffListDiff}dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Cast_1Жdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
«
Жdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/IdentityМdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackОdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
end_mask *
T0	*
Index0*
shrink_axis_mask*
new_axis_mask 
—
Еdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
“
Бdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDimsЖdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/strided_slice_2Еdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*
_output_shapes
:*
T0	*

Tdim0
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
_output_shapes
: *
dtype0
*
value	B
 Z
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
Ы
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDensednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ListDiffБdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ExpandDimsТdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesТdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
T0
*
Tindices0	
÷
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
—
~dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ReshapeReshapednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ListDiffДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Reshape/shape*
T0	*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ѕ
Бdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLike~dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Reshape*'
_output_shapes
:€€€€€€€€€*
T0	
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ў
}dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concatConcatV2~dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ReshapeБdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/zeros_likeВdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat/axis*

Tidx0*
T0	*
N*'
_output_shapes
:€€€€€€€€€
ї
|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ShapeShapednn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/ListDiff*
T0	*
out_type0*
_output_shapes
:
≠
{dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/FillFill|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Shape|dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/Const*#
_output_shapes
:€€€€€€€€€*
T0	
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
≈
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_1}dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concatДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*
T0	*'
_output_shapes
:€€€€€€€€€
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
њ
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Gather_2{dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/FillДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:€€€€€€€€€*
T0	*

Tidx0*
N
∆
Дdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorderdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_1dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/concat_2kdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Identity*
T0	*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€
Э
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/IdentityIdentitykdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Identity*
_output_shapes
:*
T0	
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ц
Иdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSliceДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseReorderОdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackРdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Рdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
new_axis_mask *#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
end_mask
ѕ
dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/CastCastИdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0	
б
Бdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/UniqueUniqueЖdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseReorder:1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Т
Лdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherCdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/readБdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/Unique*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*'
_output_shapes
:€€€€€€€€€*
Tparams0*
validate_indices(*
Tindices0	
в
zdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparseSparseSegmentMeanЛdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/embedding_lookupГdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/Unique:1dnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
√
rdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
≤
ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_1ReshapeДdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/SparseFillEmptyRows/SparseToDenserdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ґ
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/ShapeShapezdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
ј
vdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
и
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_sliceStridedSlicehdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Shapevdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stackxdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stack_1xdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice/stack_2*
end_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
_output_shapes
: *
new_axis_mask *
Index0*
T0
ђ
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
И
hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/stackPackjdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/stack/0pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
Ф
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/TileTileldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_1hdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
*

Tmultiples0
®
mdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
т
bdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweightsSelectgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Tilemdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/zeros_likezdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
М
gdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/CastCastgdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/InnerFlatten/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
є
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
с
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1Slicegdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Castpdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1/beginodnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1/size*
Index0*
T0*
_output_shapes
:
М
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Shape_1Shapebdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights*
out_type0*
_output_shapes
:*
T0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
¬
odnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
ф
jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2Slicejdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Shape_1pdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2/beginodnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2/size*
_output_shapes
:*
Index0*
T0
∞
ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ч
idnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/concatConcatV2jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_1jdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Slice_2ndnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
Ж
ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_2Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweightsidnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/concat*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
∞
ddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/ShapeShapeExpandDims_4*
T0	*
_output_shapes
:*
out_type0
Е
cdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/CastCastddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Shape*

SrcT0*
_output_shapes
:*

DstT0	
≤
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast_1/xConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ж
ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast_1Castgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0	
Ъ
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/NotEqualNotEqualExpandDims_4ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast_1*
T0	*'
_output_shapes
:€€€€€€€€€
€
ddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/WhereWheregdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/NotEqual*'
_output_shapes
:€€€€€€€€€
њ
ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
©
fdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/ReshapeReshapeExpandDims_4ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
Tshape0*
T0	
√
rdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"       
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
б
ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_sliceStridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Whererdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stacktdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice/stack_2*
shrink_axis_mask*#
_output_shapes
:€€€€€€€€€*
T0	*
Index0*
end_mask*
new_axis_mask *
ellipsis_mask *

begin_mask
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stackConst*
valueB"        *
_output_shapes
:*
dtype0
«
vdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
«
vdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
н
ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1StridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Wheretdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stackvdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1vdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2*
shrink_axis_mask *'
_output_shapes
:€€€€€€€€€*
T0	*
Index0*
end_mask*
new_axis_mask *
ellipsis_mask *

begin_mask
П
fdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/unstackUnpackcdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast*
_output_shapes
: : *

axis *	
num*
T0	
Р
ddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/stackPackhdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/unstack:1*
_output_shapes
:*
N*

axis *
T0	
с
bdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/MulMulndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slice_1ddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/stack*'
_output_shapes
:€€€€€€€€€*
T0	
Њ
tdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
О
bdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/SumSumbdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Multdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0	*#
_output_shapes
:€€€€€€€€€
й
bdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/AddAddldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/strided_slicebdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Sum*
T0	*#
_output_shapes
:€€€€€€€€€
Ч
ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/GatherGatherfdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Add*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:€€€€€€€€€
Т
Pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/mod/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ѕ
Ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/modFloorModednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/GatherPdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/mod/y*
T0	*#
_output_shapes
:€€€€€€€€€
µ
kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ї
ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_sliceStridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Castkdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stackmdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stack_1mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice/stack_2*
_output_shapes
:*
end_mask *
new_axis_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
Index0*
T0	
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
є
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
є
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
√
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1StridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Castmdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stackodnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stack_1odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1/stack_2*
_output_shapes
:*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask *
Index0*
T0	
І
]dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/ConstConst*
valueB: *
_output_shapes
:*
dtype0
к
\dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/ProdProdgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slice_1]dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
З
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat/values_1Pack\dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/Prod*
_output_shapes
:*
N*

axis *
T0	
•
cdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ў
^dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concatConcatV2ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/strided_slicegdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat/values_1cdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat/axis*
N*

Tidx0*
T0	*
_output_shapes
:
–
ednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshapeSparseReshapeddnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Wherecdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/DenseToSparseTensor/Cast^dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/concat*-
_output_shapes
:€€€€€€€€€:
ш
ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape/IdentityIdentityNdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/mod*
T0	*#
_output_shapes
:€€€€€€€€€
Е
adnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
valueB"      
ш
`dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/meanConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 
ъ
bdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
valueB
 *уµ>*
_output_shapes
: *
dtype0
Г
kdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaladnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/shape*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes

:*
T0*
dtype0*
seed2 *

seed 
≥
_dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/mulMulkdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalbdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/stddev*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes

:*
T0
°
[dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normalAdd_dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/mul`dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal/mean*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes

:*
T0
Е
>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
VariableV2*
_output_shapes

:*
dtype0*
shape
:*
	container *Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
shared_name 
С
Ednn/input_from_feature_columns/str2ex_embedding/weights/part_0/AssignAssign>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0[dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/Initializer/truncated_normal*
use_locking(*
T0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
validate_shape(*
_output_shapes

:
Л
Cdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/readIdentity>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
T0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
Є
ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
л
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SliceSlicegdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape:1ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice/beginmdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice/size*
Index0*
T0	*
_output_shapes
:
≤
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Б
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/ProdProdhdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slicehdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather/indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ю
idnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GatherGathergdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape:1qdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather/indices*
Tindices0*
validate_indices(*
Tparams0	*
_output_shapes
: 
Р
zdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/new_shapePackgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Prodidnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather*
_output_shapes
:*
N*

axis *
T0	
ь
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshapeSparseReshapeednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshapegdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape:1zdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:€€€€€€€€€:
£
ydnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/IdentityIdentityndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape/Identity*#
_output_shapes
:€€€€€€€€€*
T0	
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Ы
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GreaterEqualGreaterEqualydnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/Identityqdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
Л
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/WhereWhereodnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
√
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Н
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/ReshapeReshapehdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:€€€€€€€€€
≥
kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_1Gatherpdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshapejdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*'
_output_shapes
:€€€€€€€€€
Є
kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_2Gatherydnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape/Identityjdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:€€€€€€€€€
Р
kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/IdentityIdentityrdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseReshape:1*
_output_shapes
:*
T0	
Њ
|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
’
Кdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
њ
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/IdentityКdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackМdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0	*
Index0*
_output_shapes
: *
shrink_axis_mask
Ї
{dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/CastCastДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice*
_output_shapes
: *

DstT0*

SrcT0	
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
љ
|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/rangeRangeВdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range/start{dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/CastВdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range/delta*#
_output_shapes
:€€€€€€€€€*

Tidx0
ј
}dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Cast_1Cast|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/range*

SrcT0*#
_output_shapes
:€€€€€€€€€*

DstT0	
ё
Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
‘
Жdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_1Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackОdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*
shrink_axis_mask*#
_output_shapes
:€€€€€€€€€*
Index0*
T0	*
end_mask*
new_axis_mask *

begin_mask*
ellipsis_mask 
я
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ListDiffListDiff}dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Cast_1Жdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
out_idx0*
T0	
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
«
Жdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/IdentityМdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackОdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
end_mask *
Index0*
T0	*
shrink_axis_mask*
new_axis_mask 
—
Еdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
“
Бdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDimsЖdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/strided_slice_2Еdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
_output_shapes
:*
T0	
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
Ы
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDensednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ListDiffБdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ExpandDimsТdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesТdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
T0
*
Tindices0	
÷
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
—
~dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ReshapeReshapednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ListDiffДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Reshape/shape*
T0	*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ѕ
Бdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLike~dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:€€€€€€€€€
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
ў
}dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concatConcatV2~dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ReshapeБdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/zeros_likeВdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat/axis*
N*

Tidx0*
T0	*'
_output_shapes
:€€€€€€€€€
ї
|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ShapeShapednn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/ListDiff*
_output_shapes
:*
out_type0*
T0	
≠
{dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/FillFill|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Shape|dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:€€€€€€€€€
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
≈
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_1}dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concatДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:€€€€€€€€€*
T0	*

Tidx0*
N
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
њ
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Gather_2{dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/FillДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_2/axis*

Tidx0*
T0	*
N*#
_output_shapes
:€€€€€€€€€
∆
Дdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorderdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_1dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/concat_2kdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Identity*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
T0	
Э
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/IdentityIdentitykdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Identity*
T0	*
_output_shapes
:
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
ц
Иdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSliceДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseReorderОdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackРdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Рdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
ellipsis_mask *#
_output_shapes
:€€€€€€€€€*
end_mask*
Index0*
T0	*
shrink_axis_mask*
new_axis_mask 
ѕ
dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/CastCastИdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0	
б
Бdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/UniqueUniqueЖdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseReorder:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
out_idx0*
T0	
Т
Лdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherCdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/readБdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/Unique*'
_output_shapes
:€€€€€€€€€*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
Tparams0*
validate_indices(*
Tindices0	
в
zdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparseSparseSegmentMeanЛdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/embedding_lookupГdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/Unique:1dnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
√
rdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
≤
ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_1ReshapeДdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/SparseFillEmptyRows/SparseToDenserdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0

Ґ
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/ShapeShapezdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse*
T0*
_output_shapes
:*
out_type0
ј
vdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
и
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_sliceStridedSlicehdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Shapevdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stackxdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stack_1xdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ђ
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
И
hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/stackPackjdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/stack/0pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/strided_slice*

axis *
_output_shapes
:*
T0*
N
Ф
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/TileTileldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_1hdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
*

Tmultiples0
®
mdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
т
bdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweightsSelectgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Tilemdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/zeros_likezdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
М
gdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/CastCastgdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/InnerFlatten/SparseReshape:1*
_output_shapes
:*

DstT0*

SrcT0	
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1/beginConst*
valueB: *
_output_shapes
:*
dtype0
є
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
с
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1Slicegdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Castpdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1/beginodnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1/size*
Index0*
T0*
_output_shapes
:
М
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Shape_1Shapebdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights*
out_type0*
_output_shapes
:*
T0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
¬
odnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
ф
jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2Slicejdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Shape_1pdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2/beginodnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2/size*
Index0*
T0*
_output_shapes
:
∞
ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ч
idnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/concatConcatV2jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_1jdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Slice_2ndnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
Ж
ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_2Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweightsidnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/concat*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
∞
ddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/ShapeShapeExpandDims_6*
_output_shapes
:*
out_type0*
T0	
Е
cdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/CastCastddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Shape*

SrcT0*
_output_shapes
:*

DstT0	
≤
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast_1/xConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ж
ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast_1Castgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast_1/x*
_output_shapes
: *

DstT0	*

SrcT0
Ъ
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/NotEqualNotEqualExpandDims_6ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast_1*'
_output_shapes
:€€€€€€€€€*
T0	
€
ddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/WhereWheregdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/NotEqual*'
_output_shapes
:€€€€€€€€€
њ
ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
©
fdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/ReshapeReshapeExpandDims_6ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
Tshape0*
T0	
√
rdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stackConst*
valueB"       *
dtype0*
_output_shapes
:
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
б
ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_sliceStridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Whererdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stacktdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice/stack_2*

begin_mask*
ellipsis_mask *#
_output_shapes
:€€€€€€€€€*
end_mask*
Index0*
T0	*
shrink_axis_mask*
new_axis_mask 
≈
tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
«
vdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
«
vdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
н
ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1StridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Wheretdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stackvdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stack_1vdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0	*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask 
П
fdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/unstackUnpackcdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast*
_output_shapes
: : *

axis *	
num*
T0	
Р
ddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/stackPackhdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/unstack:1*
_output_shapes
:*
N*

axis *
T0	
с
bdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/MulMulndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slice_1ddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/stack*
T0	*'
_output_shapes
:€€€€€€€€€
Њ
tdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
О
bdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/SumSumbdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Multdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0	*#
_output_shapes
:€€€€€€€€€
й
bdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/AddAddldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/strided_slicebdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Sum*
T0	*#
_output_shapes
:€€€€€€€€€
Ч
ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/GatherGatherfdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Add*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:€€€€€€€€€
Т
Pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/mod/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
Ѕ
Ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/modFloorModednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/GatherPdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/mod/y*
T0	*#
_output_shapes
:€€€€€€€€€
µ
kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ї
ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_sliceStridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Castkdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stackmdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stack_1mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
_output_shapes
:*
end_mask *
Index0*
T0	*
shrink_axis_mask *
new_axis_mask 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
є
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
є
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
√
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1StridedSlicecdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Castmdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stackodnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stack_1odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0	
І
]dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/ConstConst*
valueB: *
dtype0*
_output_shapes
:
к
\dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/ProdProdgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slice_1]dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
З
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat/values_1Pack\dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/Prod*
_output_shapes
:*
N*

axis *
T0	
•
cdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ў
^dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concatConcatV2ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/strided_slicegdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat/values_1cdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat/axis*
_output_shapes
:*
T0	*

Tidx0*
N
–
ednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshapeSparseReshapeddnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Wherecdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/DenseToSparseTensor/Cast^dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/concat*-
_output_shapes
:€€€€€€€€€:
ш
ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape/IdentityIdentityNdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/mod*#
_output_shapes
:€€€€€€€€€*
T0	
Е
adnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
valueB"      *
_output_shapes
:*
dtype0
ш
`dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/meanConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
valueB
 *    *
_output_shapes
: *
dtype0
ъ
bdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
valueB
 *уµ>*
dtype0*
_output_shapes
: 
Г
kdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaladnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/shape*
seed2 *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*

seed *
_output_shapes

:*
T0
≥
_dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/mulMulkdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalbdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/stddev*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
_output_shapes

:*
T0
°
[dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normalAdd_dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/mul`dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal/mean*
T0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
Е
>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape
:*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
С
Ednn/input_from_feature_columns/str3ex_embedding/weights/part_0/AssignAssign>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0[dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Initializer/truncated_normal*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
_output_shapes

:*
T0*
validate_shape(*
use_locking(
Л
Cdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/readIdentity>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
T0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
Є
ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
Ј
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
л
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SliceSlicegdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape:1ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice/beginmdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice/size*
Index0*
T0	*
_output_shapes
:
≤
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Б
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/ProdProdhdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slicehdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather/indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ю
idnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GatherGathergdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape:1qdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather/indices*
_output_shapes
: *
validate_indices(*
Tparams0	*
Tindices0
Р
zdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/new_shapePackgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Prodidnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather*
T0	*

axis *
N*
_output_shapes
:
ь
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshapeSparseReshapeednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshapegdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape:1zdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:€€€€€€€€€:
£
ydnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/IdentityIdentityndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape/Identity*#
_output_shapes
:€€€€€€€€€*
T0	
≥
qdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
Ы
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GreaterEqualGreaterEqualydnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/Identityqdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
Л
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/WhereWhereodnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
√
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Н
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/ReshapeReshapehdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape/shape*
Tshape0*#
_output_shapes
:€€€€€€€€€*
T0	
≥
kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_1Gatherpdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshapejdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape*'
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0	*
Tindices0	
Є
kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_2Gatherydnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape/Identityjdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:€€€€€€€€€
Р
kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/IdentityIdentityrdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
Њ
|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
’
Кdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
њ
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/IdentityКdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stackМdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*
T0	*
Index0*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*
ellipsis_mask *

begin_mask *
end_mask 
Ї
{dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/CastCastДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice*

SrcT0	*
_output_shapes
: *

DstT0
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
љ
|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/rangeRangeВdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range/start{dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/CastВdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range/delta*#
_output_shapes
:€€€€€€€€€*

Tidx0
ј
}dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Cast_1Cast|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/range*

SrcT0*#
_output_shapes
:€€€€€€€€€*

DstT0	
ё
Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
‘
Жdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_1Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackОdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*
Index0*
T0	*
new_axis_mask *#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
end_mask
я
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ListDiffListDiff}dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Cast_1Жdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
out_idx0*
T0	
„
Мdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ў
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
«
Жdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlicekdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/IdentityМdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackОdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0	*
Index0*
_output_shapes
: *
shrink_axis_mask
—
Еdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
“
Бdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDimsЖdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/strided_slice_2Еdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*
_output_shapes
:*
T0	*

Tdim0
’
Тdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
value	B
 Z*
_output_shapes
: *
dtype0

’
Тdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
Ы
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDensednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ListDiffБdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ExpandDimsТdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesТdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
T0
*
Tindices0	
÷
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
—
~dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ReshapeReshapednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ListDiffДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Reshape/shape*
T0	*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ѕ
Бdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLike~dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:€€€€€€€€€
≈
Вdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
ў
}dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concatConcatV2~dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ReshapeБdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/zeros_likeВdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat/axis*

Tidx0*
T0	*
N*'
_output_shapes
:€€€€€€€€€
ї
|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ShapeShapednn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/ListDiff*
out_type0*
_output_shapes
:*
T0	
≠
{dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/FillFill|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Shape|dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:€€€€€€€€€
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
≈
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_1}dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concatДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:€€€€€€€€€*
T0	*

Tidx0*
N
«
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
њ
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Gather_2{dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/FillДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:€€€€€€€€€*
N*
T0	*

Tidx0
∆
Дdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorderdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_1dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/concat_2kdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Identity*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
T0	
Э
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/IdentityIdentitykdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Identity*
_output_shapes
:*
T0	
а
Оdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
в
Рdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ц
Иdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSliceДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseReorderОdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stackРdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Рdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0	*
Index0*#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask
ѕ
dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/CastCastИdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
б
Бdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/UniqueUniqueЖdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseReorder:1*
out_idx0*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Т
Лdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherCdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/readБdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/Unique*'
_output_shapes
:€€€€€€€€€*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
Tparams0*
validate_indices(*
Tindices0	
в
zdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparseSparseSegmentMeanЛdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/embedding_lookupГdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/Unique:1dnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:€€€€€€€€€
√
rdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
≤
ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_1ReshapeДdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/SparseFillEmptyRows/SparseToDenserdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ґ
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/ShapeShapezdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse*
_output_shapes
:*
out_type0*
T0
ј
vdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
¬
xdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
и
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_sliceStridedSlicehdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Shapevdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stackxdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stack_1xdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ђ
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
И
hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/stackPackjdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/stack/0pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
Ф
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/TileTileldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_1hdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
*

Tmultiples0
®
mdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
т
bdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweightsSelectgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Tilemdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/zeros_likezdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
М
gdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/CastCastgdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/InnerFlatten/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
є
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
с
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1Slicegdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Castpdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1/beginodnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1/size*
Index0*
T0*
_output_shapes
:
М
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Shape_1Shapebdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights*
T0*
out_type0*
_output_shapes
:
Ї
pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
¬
odnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
ф
jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2Slicejdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Shape_1pdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2/beginodnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2/size*
_output_shapes
:*
Index0*
T0
∞
ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ч
idnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/concatConcatV2jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_1jdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Slice_2ndnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
Ж
ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_2Reshapebdnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweightsidnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/concat*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
З
Ednn/input_from_feature_columns/input_from_feature_columns/concat/axisConst*
value	B :*
_output_shapes
: *
dtype0
ќ
@dnn/input_from_feature_columns/input_from_feature_columns/concatConcatV2ldnn/input_from_feature_columns/input_from_feature_columns/str1ex_embedding/str1ex_embeddingweights/Reshape_2ldnn/input_from_feature_columns/input_from_feature_columns/str2ex_embedding/str2ex_embeddingweights/Reshape_2ldnn/input_from_feature_columns/input_from_feature_columns/str3ex_embedding/str3ex_embeddingweights/Reshape_2ExpandDims_1ExpandDims_3ExpandDims_5Ednn/input_from_feature_columns/input_from_feature_columns/concat/axis*'
_output_shapes
:€€€€€€€€€	*
T0*

Tidx0*
N
«
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"	   
   
є
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *№њ
є
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *№?*
dtype0*
_output_shapes
: 
°
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes

:	
*
T0*
dtype0*
seed2 *

seed 
Ю
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: *
T0
∞
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes

:	

Ґ
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:	
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0
…
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
shared_name *3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
	container *
shape
:	
*
dtype0*
_output_shapes

:	

Ч
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes

:	
*
T0*
validate_shape(*
use_locking(
±
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes

:	

≤
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
dtype0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueB
*    
њ
dnn/hiddenlayer_0/biases/part_0
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
	container *
shape:
*
dtype0*
_output_shapes
:

Ж
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0
™
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
:

u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
_output_shapes

:	
*
T0
„
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
_output_shapes
:
*
T0
°
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*
data_formatNHWC*
T0*'
_output_shapes
:€€€€€€€€€

y
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€

W
zero_fraction/zeroConst*
valueB
 *    *
_output_shapes
: *
dtype0
И
zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*'
_output_shapes
:€€€€€€€€€

p
zero_fraction/CastCastzero_fraction/Equal*'
_output_shapes
:€€€€€€€€€
*

DstT0*

SrcT0

d
zero_fraction/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
Б
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ш
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values*
dtype0*
_output_shapes
: 
Я
)dnn/hiddenlayer_0_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0_fraction_of_zero_values/tagszero_fraction/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0_activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_0_activation
Щ
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
«
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"
      *
dtype0*
_output_shapes
:
є
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *:Ќњ
є
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *:Ќ?
°
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes

:
*
T0*
dtype0*
seed2 *

seed 
Ю
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
∞
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes

:

Ґ
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes

:

…
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
shared_name *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
	container *
shape
:
*
dtype0*
_output_shapes

:

Ч
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
use_locking(*
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
validate_shape(*
_output_shapes

:

±
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes

:
*
T0
≤
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueB*    *
_output_shapes
:*
dtype0
њ
dnn/hiddenlayer_1/biases/part_0
VariableV2*
_output_shapes
:*
dtype0*
shape:*
	container *2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
™
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
:
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:

ї
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *'
_output_shapes
:€€€€€€€€€*
transpose_a( *
T0
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:
°
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Y
zero_fraction_1/zeroConst*
valueB
 *    *
_output_shapes
: *
dtype0
М
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
T0*'
_output_shapes
:€€€€€€€€€
t
zero_fraction_1/CastCastzero_fraction_1/Equal*

SrcT0
*'
_output_shapes
:€€€€€€€€€*

DstT0
f
zero_fraction_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
З
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ш
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values
°
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
_output_shapes
: *
T0
}
 dnn/hiddenlayer_1_activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_1_activation
Щ
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
_output_shapes
: *
T0
«
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"      
є
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *тк-њ*
dtype0*
_output_shapes
: 
є
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *тк-?*
_output_shapes
: *
dtype0
°
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
Ю
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
∞
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes

:
Ґ
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0
…
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
shared_name *3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:
Ч
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes

:*
T0*
validate_shape(*
use_locking(
±
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
≤
1dnn/hiddenlayer_2/biases/part_0/Initializer/ConstConst*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB*    *
dtype0*
_output_shapes
:
њ
dnn/hiddenlayer_2/biases/part_0
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
Ж
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/Const*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
validate_shape(*
_output_shapes
:
™
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
:*
T0
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
T0*
_output_shapes

:
ї
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *'
_output_shapes
:€€€€€€€€€*
transpose_a( *
T0
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
:
°
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
Y
zero_fraction_2/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    
М
zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reluzero_fraction_2/zero*
T0*'
_output_shapes
:€€€€€€€€€
t
zero_fraction_2/CastCastzero_fraction_2/Equal*

SrcT0
*'
_output_shapes
:€€€€€€€€€*

DstT0
f
zero_fraction_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
З
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ш
.dnn/hiddenlayer_2_fraction_of_zero_values/tagsConst*:
value1B/ B)dnn/hiddenlayer_2_fraction_of_zero_values*
_output_shapes
: *
dtype0
°
)dnn/hiddenlayer_2_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2_fraction_of_zero_values/tagszero_fraction_2/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_2_activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_2_activation
Щ
dnn/hiddenlayer_2_activationHistogramSummary dnn/hiddenlayer_2_activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
є
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      
Ђ
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *„≥]њ*
_output_shapes
: *
dtype0
Ђ
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *„≥]?
М
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes

:*
T0*
dtype0*
seed2 *

seed 
В
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: *
T0
Ф
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes

:
Ж
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0*
T0
ї
dnn/logits/weights/part_0
VariableV2*
_output_shapes

:*
dtype0*
shape
:*
	container *,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
ы
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@dnn/logits/weights/part_0*
validate_shape(*
_output_shapes

:
Ь
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*
T0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0
§
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    
±
dnn/logits/biases/part_0
VariableV2*
shared_name *+
_class!
loc:@dnn/logits/biases/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
к
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@dnn/logits/biases/part_0*
validate_shape(*
_output_shapes
:
Х
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*
T0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
_output_shapes

:*
T0
≠
dnn/logits/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/logits/weights*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
М
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*'
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
T0
Y
zero_fraction_3/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
zero_fraction_3/EqualEqualdnn/logits/BiasAddzero_fraction_3/zero*'
_output_shapes
:€€€€€€€€€*
T0
t
zero_fraction_3/CastCastzero_fraction_3/Equal*

SrcT0
*'
_output_shapes
:€€€€€€€€€*

DstT0
f
zero_fraction_3/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
З
zero_fraction_3/MeanMeanzero_fraction_3/Castzero_fraction_3/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
К
'dnn/logits_fraction_of_zero_values/tagsConst*3
value*B( B"dnn/logits_fraction_of_zero_values*
dtype0*
_output_shapes
: 
У
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_3/Mean*
T0*
_output_shapes
: 
o
dnn/logits_activation/tagConst*&
valueB Bdnn/logits_activation*
dtype0*
_output_shapes
: 
y
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
_output_shapes
: *
T0
j
predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
_
predictions/classes/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
К
predictions/classesArgMaxdnn/logits/BiasAddpredictions/classes/dimension*#
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
О
0training_loss/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_7*
squeeze_dims
*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
.training_loss/softmax_cross_entropy_loss/ShapeShape0training_loss/softmax_cross_entropy_loss/Squeeze*
T0	*
_output_shapes
:*
out_type0
е
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
Tlabels0	
]
training_loss/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Т
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
_output_shapes
: *
T0
У
,metrics/remove_squeezable_dimensions/SqueezeSqueezeExpandDims_7*
squeeze_dims

€€€€€€€€€*
T0	*#
_output_shapes
:€€€€€€€€€
З
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*#
_output_shapes
:€€€€€€€€€*
T0	
c
metrics/ToFloatCastmetrics/Equal*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

[
metrics/accuracy/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    
z
metrics/accuracy/total
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
ћ
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
Л
metrics/accuracy/total/readIdentitymetrics/accuracy/total*
T0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
]
metrics/accuracy/zeros_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
z
metrics/accuracy/count
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
ќ
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
Л
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: *
T0
_
metrics/accuracy/SizeSizemetrics/ToFloat*
T0*
_output_shapes
: *
out_type0
i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*
_output_shapes
: *

DstT0*

SrcT0
`
metrics/accuracy/ConstConst*
valueB: *
_output_shapes
:*
dtype0
В
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
і
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*
use_locking( *
T0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
Љ
metrics/accuracy/AssignAdd_1	AssignAddmetrics/accuracy/countmetrics/accuracy/ToFloat_1*
use_locking( *
T0*
_output_shapes
: *)
_class
loc:@metrics/accuracy/count
_
metrics/accuracy/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
}
metrics/accuracy/GreaterGreatermetrics/accuracy/count/readmetrics/accuracy/Greater/y*
_output_shapes
: *
T0
~
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
_output_shapes
: *
T0
]
metrics/accuracy/value/eConst*
dtype0*
_output_shapes
: *
valueB
 *    
П
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
_output_shapes
: *
T0
a
metrics/accuracy/Greater_1/yConst*
valueB
 *    *
dtype0*
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
metrics/accuracy/update_op/eConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ы
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
_output_shapes
: *
T0
N
metrics/RankConst*
_output_shapes
: *
dtype0*
value	B :
U
metrics/LessEqual/yConst*
value	B :*
_output_shapes
: *
dtype0
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
_output_shapes
: *
T0
Т
metrics/Assert/ConstConst*
dtype0*
_output_shapes
: *N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]
Ъ
metrics/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*

T
2*
	summarize
А
metrics/Reshape/shapeConst^metrics/Assert/Assert*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
{
metrics/ReshapeReshapeExpandDims_7metrics/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
Tshape0*
T0	
]
metrics/one_hot/on_valueConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
^
metrics/one_hot/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
W
metrics/one_hot/depthConst*
value	B :*
_output_shapes
: *
dtype0
«
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*'
_output_shapes
:€€€€€€€€€*
TI0	*
axis€€€€€€€€€*
T0
f
metrics/CastCastmetrics/one_hot*'
_output_shapes
:€€€€€€€€€*

DstT0
*

SrcT0
j
metrics/auc/Reshape/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
Ф
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
l
metrics/auc/Reshape_1/shapeConst*
valueB"   €€€€*
dtype0*
_output_shapes
:
Л
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0

d
metrics/auc/ShapeShapemetrics/auc/Reshape*
out_type0*
_output_shapes
:*
T0
i
metrics/auc/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!metrics/auc/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!metrics/auc/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
µ
metrics/auc/strided_sliceStridedSlicemetrics/auc/Shapemetrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
А
metrics/auc/ConstConst*є
valueѓBђ»"†Хњ÷≥ѕ©§;ѕ©$<Јюv<ѕ©§<C‘Ќ<Јюц<Х=ѕ©$=	?9=C‘M=}ib=Јюv=ш…Е=ХР=2_Ъ=ѕ©§=lфЃ=	?є=¶Й√=C‘Ќ=аЎ=}iв=ім=Јюц=™§ >ш…>Gп
>Х>д9>2_>БД>ѕ©$>ѕ)>lф.>ї4>	?9>Wd>>¶ЙC>фЃH>C‘M>СщR>аX>.D]>}ib>ЋОg>іl>hўq>Јюv>$|>™§А>Q7Г>ш…Е>†\И>GпК>оБН>ХР><ІТ>д9Х>ЛћЧ>2_Ъ>ўсЬ>БДЯ>(Ґ>ѕ©§>v<І>ѕ©>≈aђ>lфЃ>З±>їі>bђґ>	?є>∞—ї>WdЊ>€цј>¶Й√>M∆>фЃ»>ЬAЋ>C‘Ќ>кf–>Сщ“>9М’>аЎ>З±Џ>.DЁ>÷÷я>}iв>$ьд>ЋОз>r!к>ім>ЅFп>hўс>lф>Јюц>^Сщ>$ь>ђґю>™§ ?эн?Q7?•А?ш…?L?†\?у•	?Gп
?Ъ8?оБ?BЋ?Х?й]?<І?Рр?д9?7Г?Лћ?я?2_?Ж®?ўс?-;?БД?‘Ќ ?("?{`#?ѕ©$?#у%?v<'? Е(?ѕ)?q+?≈a,?Ђ-?lф.?ј=0?З1?g–2?ї4?c5?bђ6?µх7?	?9?]И:?∞—;?=?Wd>?Ђ≠??€ц@?R@B?¶ЙC?ъ“D?MF?°eG?фЃH?HшI?ЬAK?пКL?C‘M?ЧO?кfP?>∞Q?СщR?еBT?9МU?М’V?аX?3hY?З±Z?џъ[?.D]?ВН^?÷÷_?) a?}ib?–≤c?$ьd?xEf?ЋОg?Ўh?r!j?∆jk?іl?mэm?ЅFo?Рp?hўq?Љ"s?lt?cµu?Јюv?
Hx?^Сy?≤Џz?$|?Ym}?ђґ~? А?*
dtype0*
_output_shapes	
:»
d
metrics/auc/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Й
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*
T0*
_output_shapes
:	»*

Tdim0
U
metrics/auc/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
Г
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
_output_shapes
:*
N*

axis *
T0
И
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*(
_output_shapes
:»€€€€€€€€€*
T0*

Tmultiples0
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
_output_shapes
: *
T0
]
metrics/auc/transpose/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/startConst*
value	B : *
_output_shapes
: *
dtype0
c
!metrics/auc/transpose/Range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
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
valueB"»      *
dtype0*
_output_shapes
:
Ф
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*(
_output_shapes
:»€€€€€€€€€*
T0*

Tmultiples0
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
metrics/auc/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"»      
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
metrics/auc/zerosConst*
valueB»*    *
_output_shapes	
:»*
dtype0
И
metrics/auc/true_positives
VariableV2*
_output_shapes	
:»*
	container *
dtype0*
shared_name *
shape:»
Ў
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
_output_shapes	
:»*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
T0*
use_locking(
Ь
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*
T0*
_output_shapes	
:»*-
_class#
!loc:@metrics/auc/true_positives
w
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
w
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*(
_output_shapes
:»€€€€€€€€€*

DstT0*

SrcT0

c
!metrics/auc/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
У
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:»
Ј
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*
use_locking( *
T0*
_output_shapes	
:»*-
_class#
!loc:@metrics/auc/true_positives
b
metrics/auc/zeros_1Const*
valueB»*    *
dtype0*
_output_shapes	
:»
Й
metrics/auc/false_negatives
VariableV2*
shape:»*
shared_name *
dtype0*
_output_shapes	
:»*
	container 
Ё
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes	
:»*
T0*
validate_shape(*
use_locking(
Я
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*
_output_shapes	
:»*.
_class$
" loc:@metrics/auc/false_negatives*
T0
|
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

SrcT0
*(
_output_shapes
:»€€€€€€€€€*

DstT0
e
#metrics/auc/Sum_1/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
Ч
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*
_output_shapes	
:»*
T0*
	keep_dims( *

Tidx0
љ
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*
_output_shapes	
:»*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
use_locking( 
b
metrics/auc/zeros_2Const*
dtype0*
_output_shapes	
:»*
valueB»*    
И
metrics/auc/true_negatives
VariableV2*
_output_shapes	
:»*
	container *
shape:»*
dtype0*
shared_name 
Џ
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
_output_shapes	
:»*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
use_locking(
Ь
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*
_output_shapes	
:»*-
_class#
!loc:@metrics/auc/true_negatives*
T0
В
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

SrcT0
*(
_output_shapes
:»€€€€€€€€€*

DstT0
e
#metrics/auc/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
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
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*
use_locking( *
T0*
_output_shapes	
:»*-
_class#
!loc:@metrics/auc/true_negatives
b
metrics/auc/zeros_3Const*
dtype0*
_output_shapes	
:»*
valueB»*    
Й
metrics/auc/false_positives
VariableV2*
shared_name *
dtype0*
shape:»*
_output_shapes	
:»*
	container 
Ё
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
_output_shapes	
:»*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
T0*
use_locking(
Я
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
_output_shapes	
:»*
T0

metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater*(
_output_shapes
:»€€€€€€€€€
y
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*(
_output_shapes
:»€€€€€€€€€*

DstT0*

SrcT0

e
#metrics/auc/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ч
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*
_output_shapes	
:»*
T0*
	keep_dims( *

Tidx0
љ
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*
use_locking( *
T0*
_output_shapes	
:»*.
_class$
" loc:@metrics/auc/false_positives
V
metrics/auc/add/yConst*
valueB
 *љ7Ж5*
_output_shapes
: *
dtype0
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
_output_shapes	
:»*
T0
Б
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
_output_shapes	
:»*
T0
X
metrics/auc/add_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *љ7Ж5
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
dtype0*
_output_shapes
: *
valueB
 *љ7Ж5
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
_output_shapes	
:»*
T0
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:»
k
!metrics/auc/strided_slice_1/stackConst*
valueB: *
_output_shapes
:*
dtype0
n
#metrics/auc/strided_slice_1/stack_1Const*
valueB:«*
dtype0*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
¬
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
_output_shapes	
:«*
end_mask *
T0*
Index0*
shrink_axis_mask *
new_axis_mask 
k
!metrics/auc/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
m
#metrics/auc/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¬
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
end_mask*
ellipsis_mask *

begin_mask *
shrink_axis_mask *
_output_shapes	
:«*
new_axis_mask *
T0*
Index0
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
_output_shapes	
:«*
T0
k
!metrics/auc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:«
m
#metrics/auc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ј
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
end_mask *
ellipsis_mask *

begin_mask*
shrink_axis_mask *
_output_shapes	
:«*
new_axis_mask *
T0*
Index0
k
!metrics/auc/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
m
#metrics/auc/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ј
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes	
:«*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
end_mask
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:«
Z
metrics/auc/truediv/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
_output_shapes	
:«*
T0
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:«
]
metrics/auc/Const_1Const*
valueB: *
dtype0*
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
dtype0*
_output_shapes
: *
valueB
 *љ7Ж5
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
_output_shapes	
:»*
T0
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
_output_shapes	
:»*
T0
X
metrics/auc/add_8/yConst*
valueB
 *љ7Ж5*
dtype0*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
_output_shapes	
:»*
T0
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
_output_shapes	
:»*
T0
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:»
Y
metrics/auc/add_10/yConst*
valueB
 *љ7Ж5*
dtype0*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:»
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
_output_shapes	
:»*
T0
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB: 
n
#metrics/auc/strided_slice_5/stack_1Const*
valueB:«*
_output_shapes
:*
dtype0
m
#metrics/auc/strided_slice_5/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
¬
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
shrink_axis_mask *
Index0*
T0*
end_mask *
_output_shapes	
:«*

begin_mask*
ellipsis_mask 
k
!metrics/auc/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
m
#metrics/auc/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¬
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes	
:«
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
_output_shapes	
:«*
T0
k
!metrics/auc/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
valueB:«*
_output_shapes
:*
dtype0
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
¬
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
ellipsis_mask *

begin_mask*
_output_shapes	
:«*
end_mask *
T0*
Index0*
shrink_axis_mask *
new_axis_mask 
k
!metrics/auc/strided_slice_8/stackConst*
valueB:*
_output_shapes
:*
dtype0
m
#metrics/auc/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
m
#metrics/auc/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¬
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
ellipsis_mask *

begin_mask *
_output_shapes	
:«*
end_mask*
T0*
Index0*
shrink_axis_mask *
new_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
_output_shapes	
:«*
T0
\
metrics/auc/truediv_1/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
s
metrics/auc/truediv_1RealDivmetrics/auc/add_11metrics/auc/truediv_1/y*
_output_shapes	
:«*
T0
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
_output_shapes	
:«*
T0
]
metrics/auc/Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
В
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
И
*metrics/softmax_cross_entropy_loss/SqueezeSqueezeExpandDims_7*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims

Т
(metrics/softmax_cross_entropy_loss/ShapeShape*metrics/softmax_cross_entropy_loss/Squeeze*
T0	*
_output_shapes
:*
out_type0
ў
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*6
_output_shapes$
":€€€€€€€€€:€€€€€€€€€*
Tlabels0	*
T0
a
metrics/eval_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ф
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
W
metrics/mean/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    
v
metrics/mean/total
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
Љ
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*%
_class
loc:@metrics/mean/total*
_output_shapes
: *
T0*
validate_shape(*
use_locking(

metrics/mean/total/readIdentitymetrics/mean/total*
T0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
v
metrics/mean/count
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
Њ
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*%
_class
loc:@metrics/mean/count*
_output_shapes
: *
T0*
validate_shape(*
use_locking(

metrics/mean/count/readIdentitymetrics/mean/count*
T0*
_output_shapes
: *%
_class
loc:@metrics/mean/count
S
metrics/mean/SizeConst*
_output_shapes
: *
dtype0*
value	B :
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*
_output_shapes
: *

DstT0*

SrcT0
U
metrics/mean/ConstConst*
dtype0*
_output_shapes
: *
valueB 
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
loc:@metrics/mean/total*
_output_shapes
: *
T0*
use_locking( 
ђ
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*
use_locking( *
T0*
_output_shapes
: *%
_class
loc:@metrics/mean/count
[
metrics/mean/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
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
metrics/mean/value/eConst*
_output_shapes
: *
dtype0*
valueB
 *    

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
_output_shapes
: *
T0
]
metrics/mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
_output_shapes
: *
T0
]
metrics/mean/update_op/eConst*
valueB
 *    *
_output_shapes
: *
dtype0
Л
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
`

group_depsNoOp^metrics/mean/update_op^metrics/auc/update_op^metrics/accuracy/update_op
\
eval_step/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
m
	eval_step
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
¶
eval_step/AssignAssign	eval_stepeval_step/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@eval_step*
T0*
use_locking(
d
eval_step/readIdentity	eval_step*
T0*
_output_shapes
: *
_class
loc:@eval_step
T
AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@eval_step
Ј
initNoOp^global_step/AssignF^dnn/input_from_feature_columns/str1ex_embedding/weights/part_0/AssignF^dnn/input_from_feature_columns/str2ex_embedding/weights/part_0/AssignF^dnn/input_from_feature_columns/str3ex_embedding/weights/part_0/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign(^dnn/hiddenlayer_2/weights/part_0/Assign'^dnn/hiddenlayer_2/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step
З
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
З
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitialized>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
З
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
Ћ
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
…
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
dtype0*
_output_shapes
: 
Ћ
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
…
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: *
dtype0
Ћ
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
dtype0*
_output_shapes
: 
…
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
Њ
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializeddnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: *
dtype0
Љ
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializeddnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
dtype0*
_output_shapes
: 
ъ
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized7read_batch_features/file_name_queue/limit_epochs/epochs*J
_class@
><loc:@read_batch_features/file_name_queue/limit_epochs/epochs*
dtype0	*
_output_shapes
: 
Є
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedmetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: *
dtype0
Є
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedmetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: *
dtype0
ј
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedmetrics/auc/true_positives*
dtype0*
_output_shapes
: *-
_class#
!loc:@metrics/auc/true_positives
¬
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedmetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes
: *
dtype0
ј
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedmetrics/auc/true_negatives*
dtype0*
_output_shapes
: *-
_class#
!loc:@metrics/auc/true_negatives
¬
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedmetrics/auc/false_positives*
dtype0*
_output_shapes
: *.
_class$
" loc:@metrics/auc/false_positives
∞
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedmetrics/mean/total*%
_class
loc:@metrics/mean/total*
_output_shapes
: *
dtype0
∞
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitializedmetrics/mean/count*
_output_shapes
: *
dtype0*%
_class
loc:@metrics/mean/count
Ю
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized	eval_step*
_class
loc:@eval_step*
_output_shapes
: *
dtype0
ј

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_21*
T0
*

axis *
N*
_output_shapes
:
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
«
$report_uninitialized_variables/ConstConst*о
valueдBбBglobal_stepB>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0B>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0B>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B7read_batch_features/file_name_queue/limit_epochs/epochsBmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/countB	eval_step*
_output_shapes
:*
dtype0
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ў
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
_output_shapes
:*
end_mask *
new_axis_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
Index0*
T0
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
х
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
б
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
end_mask*
T0*
Index0*
shrink_axis_mask *
new_axis_mask 
ѓ
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
_output_shapes
:*
N*

axis *
T0
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ђ
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
Ћ
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
_output_shapes
:*
Tshape0*
T0
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
џ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:*
T0

Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:€€€€€€€€€
ґ
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims

В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0*
Tindices0	
g
$report_uninitialized_resources/ConstConst*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*#
_output_shapes
:€€€€€€€€€*
N*
T0*

Tidx0
°
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_output_shapes
: *
dtype0	*
_class
loc:@global_step
Й
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0
Й
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitialized>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0
Й
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0*
_output_shapes
: *
dtype0
Ќ
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
dtype0*
_output_shapes
: 
Ћ
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0
Ќ
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
Ћ
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0
Ќ
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: *
dtype0
Ћ
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
dtype0*
_output_shapes
: 
ј
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitializeddnn/logits/weights/part_0*
dtype0*
_output_shapes
: *,
_class"
 loc:@dnn/logits/weights/part_0
Њ
9report_uninitialized_variables_1/IsVariableInitialized_11IsVariableInitializeddnn/logits/biases/part_0*
_output_shapes
: *
dtype0*+
_class!
loc:@dnn/logits/biases/part_0
†
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_109report_uninitialized_variables_1/IsVariableInitialized_11*
T0
*

axis *
N*
_output_shapes
:
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
ї
&report_uninitialized_variables_1/ConstConst*а
value÷B”Bglobal_stepB>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0B>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0B>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0*
dtype0*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
г
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes
:*
shrink_axis_mask 
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
ы
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
л
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
≥
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
T0*

axis *
N*
_output_shapes
:
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
≥
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
—
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
T0*
Tshape0*
_output_shapes
:
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
б
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
T0
*
Tshape0*
_output_shapes
:
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
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*#
_output_shapes
:€€€€€€€€€*
validate_indices(*
Tparams0*
Tindices0	
м
init_2NoOp?^read_batch_features/file_name_queue/limit_epochs/epochs/Assign^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign^eval_step/Assign

init_all_tablesNoOp
/
group_deps_2NoOp^init_2^init_all_tables
ї
Merge/MergeSummaryMergeSummary7read_batch_features/file_name_queue/fraction_of_32_full)read_batch_features/fraction_of_2000_full_read_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full)dnn/hiddenlayer_0_fraction_of_zero_valuesdnn/hiddenlayer_0_activation)dnn/hiddenlayer_1_fraction_of_zero_valuesdnn/hiddenlayer_1_activation)dnn/hiddenlayer_2_fraction_of_zero_valuesdnn/hiddenlayer_2_activation"dnn/logits_fraction_of_zero_valuesdnn/logits_activationtraining_loss/ScalarSummary*
_output_shapes
: *
N
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_369eff4c4b72430cb3790865237e6ed1/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
а
save/SaveV2/tensor_namesConst*У
valueЙBЖBdnn/hiddenlayer_0/biasesBdnn/hiddenlayer_0/weightsBdnn/hiddenlayer_1/biasesBdnn/hiddenlayer_1/weightsBdnn/hiddenlayer_2/biasesBdnn/hiddenlayer_2/weightsB7dnn/input_from_feature_columns/str1ex_embedding/weightsB7dnn/input_from_feature_columns/str2ex_embedding/weightsB7dnn/input_from_feature_columns/str3ex_embedding/weightsBdnn/logits/biasesBdnn/logits/weightsBglobal_step*
dtype0*
_output_shapes
:
е
save/SaveV2/shape_and_slicesConst*Ф
valueКBЗB10 0,10B9 10 0,9:0,10B8 0,8B10 8 0,10:0,8B5 0,5B8 5 0,8:0,5B9 2 0,9:0,2B8 2 0,8:0,2B8 2 0,8:0,2B3 0,3B5 3 0,5:0,3B *
_output_shapes
:*
dtype0
Б
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read%dnn/hiddenlayer_0/weights/part_0/read$dnn/hiddenlayer_1/biases/part_0/read%dnn/hiddenlayer_1/weights/part_0/read$dnn/hiddenlayer_2/biases/part_0/read%dnn/hiddenlayer_2/weights/part_0/readCdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/readCdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/readCdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/readdnn/logits/biases/part_0/readdnn/logits/weights/part_0/readglobal_step*
dtypes
2	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
_output_shapes
: *
T0
|
save/RestoreV2/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_0/biases*
dtype0*
_output_shapes
:
o
save/RestoreV2/shape_and_slicesConst*
valueBB10 0,10*
dtype0*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
»
save/AssignAssigndnn/hiddenlayer_0/biases/part_0save/RestoreV2*
_output_shapes
:
*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
use_locking(

save/RestoreV2_1/tensor_namesConst*.
value%B#Bdnn/hiddenlayer_0/weights*
dtype0*
_output_shapes
:
w
!save/RestoreV2_1/shape_and_slicesConst*"
valueBB9 10 0,9:0,10*
_output_shapes
:*
dtype0
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
“
save/Assign_1Assign dnn/hiddenlayer_0/weights/part_0save/RestoreV2_1*
use_locking(*
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
validate_shape(*
_output_shapes

:	

~
save/RestoreV2_2/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_1/biases*
dtype0*
_output_shapes
:
o
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueBB8 0,8
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
ћ
save/Assign_2Assigndnn/hiddenlayer_1/biases/part_0save/RestoreV2_2*
_output_shapes
:*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
use_locking(

save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*.
value%B#Bdnn/hiddenlayer_1/weights
w
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*"
valueBB10 8 0,10:0,8
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
“
save/Assign_3Assign dnn/hiddenlayer_1/weights/part_0save/RestoreV2_3*
use_locking(*
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
validate_shape(*
_output_shapes

:

~
save/RestoreV2_4/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_2/biases*
dtype0*
_output_shapes
:
o
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueBB5 0,5
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
ћ
save/Assign_4Assigndnn/hiddenlayer_2/biases/part_0save/RestoreV2_4*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

save/RestoreV2_5/tensor_namesConst*.
value%B#Bdnn/hiddenlayer_2/weights*
_output_shapes
:*
dtype0
u
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
dtype0* 
valueBB8 5 0,8:0,5
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
“
save/Assign_5Assign dnn/hiddenlayer_2/weights/part_0save/RestoreV2_5*
_output_shapes

:*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
use_locking(
Э
save/RestoreV2_6/tensor_namesConst*
dtype0*
_output_shapes
:*L
valueCBAB7dnn/input_from_feature_columns/str1ex_embedding/weights
u
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
dtype0* 
valueBB9 2 0,9:0,2
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
О
save/Assign_6Assign>dnn/input_from_feature_columns/str1ex_embedding/weights/part_0save/RestoreV2_6*
_output_shapes

:	*
validate_shape(*Q
_classG
ECloc:@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0*
T0*
use_locking(
Э
save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*
dtype0*L
valueCBAB7dnn/input_from_feature_columns/str2ex_embedding/weights
u
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
_output_shapes
:* 
valueBB8 2 0,8:0,2
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
О
save/Assign_7Assign>dnn/input_from_feature_columns/str2ex_embedding/weights/part_0save/RestoreV2_7*Q
_classG
ECloc:@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0*
_output_shapes

:*
T0*
validate_shape(*
use_locking(
Э
save/RestoreV2_8/tensor_namesConst*
dtype0*
_output_shapes
:*L
valueCBAB7dnn/input_from_feature_columns/str3ex_embedding/weights
u
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
dtype0* 
valueBB8 2 0,8:0,2
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
О
save/Assign_8Assign>dnn/input_from_feature_columns/str3ex_embedding/weights/part_0save/RestoreV2_8*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*Q
_classG
ECloc:@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0
w
save/RestoreV2_9/tensor_namesConst*&
valueBBdnn/logits/biases*
dtype0*
_output_shapes
:
o
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueBB3 0,3
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
Њ
save/Assign_9Assigndnn/logits/biases/part_0save/RestoreV2_9*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0
y
save/RestoreV2_10/tensor_namesConst*'
valueBBdnn/logits/weights*
dtype0*
_output_shapes
:
v
"save/RestoreV2_10/shape_and_slicesConst* 
valueBB5 3 0,5:0,3*
_output_shapes
:*
dtype0
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
∆
save/Assign_10Assigndnn/logits/weights/part_0save/RestoreV2_10*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0
r
save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBglobal_step
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
Щ
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2	
Ґ
save/Assign_11Assignglobal_stepsave/RestoreV2_11*
_output_shapes
: *
validate_shape(*
_class
loc:@global_step*
T0	*
use_locking(
Џ
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11
-
save/restore_allNoOp^save/restore_shard""
init_op

group_deps_1"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0" 
global_step

global_step:0"я
dnn„
‘
@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0:0
@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0:0
@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0:0
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0"є
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
eval_step:0"!
local_init_op

group_deps_2"”
	variables≈¬
7
global_step:0global_step/Assignglobal_step/read:0
Ч
@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0:0Ednn/input_from_feature_columns/str1ex_embedding/weights/part_0/AssignEdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/read:0"E
7dnn/input_from_feature_columns/str1ex_embedding/weights	  "	
Ч
@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0:0Ednn/input_from_feature_columns/str2ex_embedding/weights/part_0/AssignEdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/read:0"E
7dnn/input_from_feature_columns/str2ex_embedding/weights  "
Ч
@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0:0Ednn/input_from_feature_columns/str3ex_embedding/weights/part_0/AssignEdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/read:0"E
7dnn/input_from_feature_columns/str3ex_embedding/weights  "
Я
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weights	
  "	

Ш
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biases
 "

Я
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weights
  "

Ш
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biases "
Я
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weights  "
Ш
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases "
Г
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases ""&

summary_op

Merge/MergeSummary:0"
	eval_step

eval_step:0"л
model_variables„
‘
@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0:0
@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0:0
@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0:0
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0"П#
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
+read_batch_features/read/ReaderReadUpToV2:1i
+read_batch_features/read/ReaderReadUpToV2:1:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_2:1i
+read_batch_features/read/ReaderReadUpToV2:0:read_batch_features/cond/fifo_queue_EnqueueMany/Switch_1:1\
 read_batch_features/fifo_queue:08read_batch_features/cond/fifo_queue_EnqueueMany/Switch:1
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
-read_batch_features/read/ReaderReadUpToV2_1:1m
-read_batch_features/read/ReaderReadUpToV2_1:1<read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch_2:1^
 read_batch_features/fifo_queue:0:read_batch_features/cond_1/fifo_queue_EnqueueMany/Switch:1m
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
-read_batch_features/read/ReaderReadUpToV2_2:1m
-read_batch_features/read/ReaderReadUpToV2_2:0<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_1:1m
-read_batch_features/read/ReaderReadUpToV2_2:1<read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch_2:1^
 read_batch_features/fifo_queue:0:read_batch_features/cond_2/fifo_queue_EnqueueMany/Switch:1
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
-read_batch_features/read/ReaderReadUpToV2_3:1m
-read_batch_features/read/ReaderReadUpToV2_3:1<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_2:1m
-read_batch_features/read/ReaderReadUpToV2_3:0<read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch_1:1^
 read_batch_features/fifo_queue:0:read_batch_features/cond_3/fifo_queue_EnqueueMany/Switch:1
ш
&read_batch_features/cond_3/cond_text_1$read_batch_features/cond_3/pred_id:0%read_batch_features/cond_3/switch_f:0*А
1read_batch_features/cond_3/control_dependency_1:0
$read_batch_features/cond_3/pred_id:0
%read_batch_features/cond_3/switch_f:0"
ready_op


concat:0"°
	summariesУ
Р
9read_batch_features/file_name_queue/fraction_of_32_full:0
+read_batch_features/fraction_of_2000_full:0
aread_batch_features/queue/parsed_features/read_batch_features/fifo_queue_1/fraction_of_100_full:0
+dnn/hiddenlayer_0_fraction_of_zero_values:0
dnn/hiddenlayer_0_activation:0
+dnn/hiddenlayer_1_fraction_of_zero_values:0
dnn/hiddenlayer_1_activation:0
+dnn/hiddenlayer_2_fraction_of_zero_values:0
dnn/hiddenlayer_2_activation:0
$dnn/logits_fraction_of_zero_values:0
dnn/logits_activation:0
training_loss/ScalarSummary:0"§
trainable_variablesМЙ
Ч
@dnn/input_from_feature_columns/str1ex_embedding/weights/part_0:0Ednn/input_from_feature_columns/str1ex_embedding/weights/part_0/AssignEdnn/input_from_feature_columns/str1ex_embedding/weights/part_0/read:0"E
7dnn/input_from_feature_columns/str1ex_embedding/weights	  "	
Ч
@dnn/input_from_feature_columns/str2ex_embedding/weights/part_0:0Ednn/input_from_feature_columns/str2ex_embedding/weights/part_0/AssignEdnn/input_from_feature_columns/str2ex_embedding/weights/part_0/read:0"E
7dnn/input_from_feature_columns/str2ex_embedding/weights  "
Ч
@dnn/input_from_feature_columns/str3ex_embedding/weights/part_0:0Ednn/input_from_feature_columns/str3ex_embedding/weights/part_0/AssignEdnn/input_from_feature_columns/str3ex_embedding/weights/part_0/read:0"E
7dnn/input_from_feature_columns/str3ex_embedding/weights  "
Я
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weights	
  "	

Ш
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biases
 "

Я
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weights
  "

Ш
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biases "
Я
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weights  "
Ш
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases "
Г
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases ""J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"ћ
queue_runnersЇЈ
б
#read_batch_features/file_name_queue?read_batch_features/file_name_queue/file_name_queue_EnqueueMany9read_batch_features/file_name_queue/file_name_queue_Close";read_batch_features/file_name_queue/file_name_queue_Close_1*
€
read_batch_features/fifo_queue read_batch_features/cond/Merge:0"read_batch_features/cond_1/Merge:0"read_batch_features/cond_2/Merge:0"read_batch_features/cond_3/Merge:0$read_batch_features/fifo_queue_Close"&read_batch_features/fifo_queue_Close_1*
ќ
 read_batch_features/fifo_queue_1(read_batch_features/fifo_queue_1_enqueue*read_batch_features/fifo_queue_1_enqueue_1&read_batch_features/fifo_queue_1_Close"(read_batch_features/fifo_queue_1_Close_1*(ВыF       r5є•	вк.K°9÷A*9

loss;о?


auc<,м>

global_step

accuracyЪЩЩ>h?T