       �K"	   �7�Abrain.Event:2���~�     �?p	K��7�A"�


global_step/Initializer/ConstConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
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
�
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
T0	*
_class
loc:@global_step*
_output_shapes
: 
}
input_producer/ConstConst*5
value,B*B /tmp/tmp8TBLUm/eval_csv_data.csv*
dtype0*
_output_shapes
:
U
input_producer/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Z
input_producer/Greater/yConst*
value	B : *
_output_shapes
: *
dtype0
q
input_producer/GreaterGreaterinput_producer/Sizeinput_producer/Greater/y*
T0*
_output_shapes
: 
�
input_producer/Assert/ConstConst*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: *
dtype0
�
#input_producer/Assert/Assert/data_0Const*G
value>B< B6string_input_producer requires a non-null input tensor*
dtype0*
_output_shapes
: 
�
input_producer/Assert/AssertAssertinput_producer/Greater#input_producer/Assert/Assert/data_0*

T
2*
	summarize
}
input_producer/IdentityIdentityinput_producer/Const^input_producer/Assert/Assert*
T0*
_output_shapes
:
c
!input_producer/limit_epochs/ConstConst*
value	B	 R *
_output_shapes
: *
dtype0	
�
"input_producer/limit_epochs/epochs
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
�
)input_producer/limit_epochs/epochs/AssignAssign"input_producer/limit_epochs/epochs!input_producer/limit_epochs/Const*
use_locking(*
T0	*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
validate_shape(*
_output_shapes
: 
�
'input_producer/limit_epochs/epochs/readIdentity"input_producer/limit_epochs/epochs*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
_output_shapes
: *
T0	
�
%input_producer/limit_epochs/CountUpTo	CountUpTo"input_producer/limit_epochs/epochs*
T0	*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
_output_shapes
: *
limit
�
input_producer/limit_epochsIdentityinput_producer/Identity&^input_producer/limit_epochs/CountUpTo*
_output_shapes
:*
T0
�
input_producerFIFOQueueV2*
shapes
: *
_output_shapes
: *
component_types
2*
shared_name *
	container *
capacity 
�
)input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producerinput_producer/limit_epochs*
Tcomponents
2*

timeout_ms���������
b
#input_producer/input_producer_CloseQueueCloseV2input_producer*
cancel_pending_enqueues( 
d
%input_producer/input_producer_Close_1QueueCloseV2input_producer*
cancel_pending_enqueues(
Y
"input_producer/input_producer_SizeQueueSizeV2input_producer*
_output_shapes
: 
o
input_producer/CastCast"input_producer/input_producer_Size*

SrcT0*
_output_shapes
: *

DstT0
Y
input_producer/mul/yConst*
valueB
 *   =*
dtype0*
_output_shapes
: 
e
input_producer/mulMulinput_producer/Castinput_producer/mul/y*
T0*
_output_shapes
: 
�
'input_producer/fraction_of_32_full/tagsConst*3
value*B( B"input_producer/fraction_of_32_full*
dtype0*
_output_shapes
: 
�
"input_producer/fraction_of_32_fullScalarSummary'input_producer/fraction_of_32_full/tagsinput_producer/mul*
T0*
_output_shapes
: 
y
TextLineReaderV2TextLineReaderV2*
skip_header_lines *
shared_name *
_output_shapes
: *
	container 
^
ReaderReadUpToV2/num_recordsConst*
value	B	 R
*
_output_shapes
: *
dtype0	
�
ReaderReadUpToV2ReaderReadUpToV2TextLineReaderV2input_producerReaderReadUpToV2/num_records*2
_output_shapes 
:���������:���������
M
batch/ConstConst*
value	B
 Z*
_output_shapes
: *
dtype0

�
batch/fifo_queueFIFOQueueV2*
shapes
: : *
shared_name *
capacity�*
	container *
_output_shapes
: *
component_types
2
X
batch/cond/SwitchSwitchbatch/Constbatch/Const*
_output_shapes
: : *
T0

U
batch/cond/switch_tIdentitybatch/cond/Switch:1*
_output_shapes
: *
T0

S
batch/cond/switch_fIdentitybatch/cond/Switch*
_output_shapes
: *
T0

L
batch/cond/pred_idIdentitybatch/Const*
T0
*
_output_shapes
: 
�
(batch/cond/fifo_queue_EnqueueMany/SwitchSwitchbatch/fifo_queuebatch/cond/pred_id*#
_class
loc:@batch/fifo_queue*
_output_shapes
: : *
T0
�
*batch/cond/fifo_queue_EnqueueMany/Switch_1SwitchReaderReadUpToV2batch/cond/pred_id*
T0*#
_class
loc:@ReaderReadUpToV2*2
_output_shapes 
:���������:���������
�
*batch/cond/fifo_queue_EnqueueMany/Switch_2SwitchReaderReadUpToV2:1batch/cond/pred_id*#
_class
loc:@ReaderReadUpToV2*2
_output_shapes 
:���������:���������*
T0
�
!batch/cond/fifo_queue_EnqueueManyQueueEnqueueManyV2*batch/cond/fifo_queue_EnqueueMany/Switch:1,batch/cond/fifo_queue_EnqueueMany/Switch_1:1,batch/cond/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms���������
�
batch/cond/control_dependencyIdentitybatch/cond/switch_t"^batch/cond/fifo_queue_EnqueueMany*
T0
*&
_class
loc:@batch/cond/switch_t*
_output_shapes
: 
-
batch/cond/NoOpNoOp^batch/cond/switch_f
�
batch/cond/control_dependency_1Identitybatch/cond/switch_f^batch/cond/NoOp*&
_class
loc:@batch/cond/switch_f*
_output_shapes
: *
T0

�
batch/cond/MergeMergebatch/cond/control_dependency_1batch/cond/control_dependency*
T0
*
N*
_output_shapes
: : 
W
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue*
cancel_pending_enqueues( 
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*
_output_shapes
: *

DstT0*

SrcT0
P
batch/mul/yConst*
valueB
 *t�;*
dtype0*
_output_shapes
: 
J
	batch/mulMul
batch/Castbatch/mul/y*
T0*
_output_shapes
: 
z
batch/fraction_of_150_full/tagsConst*+
value"B  Bbatch/fraction_of_150_full*
_output_shapes
: *
dtype0
x
batch/fraction_of_150_fullScalarSummarybatch/fraction_of_150_full/tags	batch/mul*
_output_shapes
: *
T0
I
batch/nConst*
value	B :
*
dtype0*
_output_shapes
: 
�
batchQueueDequeueManyV2batch/fifo_queuebatch/n*

timeout_ms���������* 
_output_shapes
:
:
*
component_types
2
i
 csv_to_tensors/record_defaults_0Const*
valueB
B *
_output_shapes
:*
dtype0
i
 csv_to_tensors/record_defaults_1Const*
valueB
B *
_output_shapes
:*
dtype0
m
 csv_to_tensors/record_defaults_2Const*
valueB*��sA*
_output_shapes
:*
dtype0
m
 csv_to_tensors/record_defaults_3Const*
valueB*�pA*
_output_shapes
:*
dtype0
m
 csv_to_tensors/record_defaults_4Const*
dtype0*
_output_shapes
:*
valueB*ײ�@
i
 csv_to_tensors/record_defaults_5Const*
_output_shapes
:*
dtype0*
valueB
B 
i
 csv_to_tensors/record_defaults_6Const*
_output_shapes
:*
dtype0*
valueB
B 
i
 csv_to_tensors/record_defaults_7Const*
_output_shapes
:*
dtype0*
valueB
B 
�
csv_to_tensors	DecodeCSVbatch:1 csv_to_tensors/record_defaults_0 csv_to_tensors/record_defaults_1 csv_to_tensors/record_defaults_2 csv_to_tensors/record_defaults_3 csv_to_tensors/record_defaults_4 csv_to_tensors/record_defaults_5 csv_to_tensors/record_defaults_6 csv_to_tensors/record_defaults_7*D
_output_shapes2
0:
:
:
:
:
:
:
:
*
field_delim,*
OUT_TYPE

2
P
ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
m

ExpandDims
ExpandDimscsv_to_tensorsExpandDims/dim*
T0*
_output_shapes

:
*

Tdim0
R
ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
s
ExpandDims_1
ExpandDimscsv_to_tensors:1ExpandDims_1/dim*
T0*
_output_shapes

:
*

Tdim0
R
ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B :
s
ExpandDims_2
ExpandDimscsv_to_tensors:2ExpandDims_2/dim*
T0*
_output_shapes

:
*

Tdim0
R
ExpandDims_3/dimConst*
dtype0*
_output_shapes
: *
value	B :
s
ExpandDims_3
ExpandDimscsv_to_tensors:3ExpandDims_3/dim*
_output_shapes

:
*
T0*

Tdim0
R
ExpandDims_4/dimConst*
dtype0*
_output_shapes
: *
value	B :
s
ExpandDims_4
ExpandDimscsv_to_tensors:4ExpandDims_4/dim*
_output_shapes

:
*
T0*

Tdim0
R
ExpandDims_5/dimConst*
dtype0*
_output_shapes
: *
value	B :
s
ExpandDims_5
ExpandDimscsv_to_tensors:5ExpandDims_5/dim*
T0*
_output_shapes

:
*

Tdim0
R
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :
s
ExpandDims_6
ExpandDimscsv_to_tensors:6ExpandDims_6/dim*
T0*
_output_shapes

:
*

Tdim0
R
ExpandDims_7/dimConst*
dtype0*
_output_shapes
: *
value	B :
s
ExpandDims_7
ExpandDimscsv_to_tensors:7ExpandDims_7/dim*
T0*
_output_shapes

:
*

Tdim0
g
"numerical_feature_preprocess/Sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *�.<
�
 numerical_feature_preprocess/SubSubExpandDims_2"numerical_feature_preprocess/Sub/y*
T0*
_output_shapes

:

g
"numerical_feature_preprocess/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
�
 numerical_feature_preprocess/mulMul numerical_feature_preprocess/Sub"numerical_feature_preprocess/Const*
T0*
_output_shapes

:

i
$numerical_feature_preprocess/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *��A
�
$numerical_feature_preprocess/truedivRealDiv numerical_feature_preprocess/mul$numerical_feature_preprocess/Const_1*
T0*
_output_shapes

:

i
$numerical_feature_preprocess/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ��
�
 numerical_feature_preprocess/addAdd$numerical_feature_preprocess/truediv$numerical_feature_preprocess/Const_2*
T0*
_output_shapes

:

i
$numerical_feature_preprocess/Sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
"numerical_feature_preprocess/Sub_1SubExpandDims_3$numerical_feature_preprocess/Sub_1/y*
_output_shapes

:
*
T0
i
$numerical_feature_preprocess/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *   A
�
"numerical_feature_preprocess/mul_1Mul"numerical_feature_preprocess/Sub_1$numerical_feature_preprocess/Const_3*
_output_shapes

:
*
T0
i
$numerical_feature_preprocess/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *  �A
�
&numerical_feature_preprocess/truediv_1RealDiv"numerical_feature_preprocess/mul_1$numerical_feature_preprocess/Const_4*
T0*
_output_shapes

:

i
$numerical_feature_preprocess/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *  ��
�
"numerical_feature_preprocess/add_1Add&numerical_feature_preprocess/truediv_1$numerical_feature_preprocess/Const_5*
T0*
_output_shapes

:

�
/target_feature_preprocess/string_to_index/ConstConst*
_output_shapes
:*
dtype0*"
valueBB102B100B101
p
.target_feature_preprocess/string_to_index/SizeConst*
_output_shapes
: *
dtype0*
value	B :
w
5target_feature_preprocess/string_to_index/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
w
5target_feature_preprocess/string_to_index/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
/target_feature_preprocess/string_to_index/rangeRange5target_feature_preprocess/string_to_index/range/start.target_feature_preprocess/string_to_index/Size5target_feature_preprocess/string_to_index/range/delta*

Tidx0*
_output_shapes
:
�
.target_feature_preprocess/string_to_index/CastCast/target_feature_preprocess/string_to_index/range*

SrcT0*
_output_shapes
:*

DstT0	
�
4target_feature_preprocess/string_to_index/hash_table	HashTable*
_output_shapes
:*
value_dtype0	*
	container *
	key_dtype0*
use_node_name_sharing( *
shared_name 
�
:target_feature_preprocess/string_to_index/hash_table/ConstConst*
dtype0	*
_output_shapes
: *
valueB	 R
���������
�
?target_feature_preprocess/string_to_index/hash_table/table_initInitializeTable4target_feature_preprocess/string_to_index/hash_table/target_feature_preprocess/string_to_index/Const.target_feature_preprocess/string_to_index/Cast*

Tkey0*

Tval0	*G
_class=
;9loc:@target_feature_preprocess/string_to_index/hash_table
�
+target_feature_preprocess/hash_table_LookupLookupTableFind4target_feature_preprocess/string_to_index/hash_tableExpandDims_1:target_feature_preprocess/string_to_index/hash_table/Const*

Tout0	*
_output_shapes

:
*	
Tin0*G
_class=
;9loc:@target_feature_preprocess/string_to_index/hash_table
�
4categorical_feature_preprocess/string_to_index/ConstConst*
_output_shapes
:*
dtype0*C
value:B8BblueBbrownByellowBpinkBblackBgreenBredB 
u
3categorical_feature_preprocess/string_to_index/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
:categorical_feature_preprocess/string_to_index/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
|
:categorical_feature_preprocess/string_to_index/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
4categorical_feature_preprocess/string_to_index/rangeRange:categorical_feature_preprocess/string_to_index/range/start3categorical_feature_preprocess/string_to_index/Size:categorical_feature_preprocess/string_to_index/range/delta*
_output_shapes
:*

Tidx0
�
3categorical_feature_preprocess/string_to_index/CastCast4categorical_feature_preprocess/string_to_index/range*
_output_shapes
:*

DstT0	*

SrcT0
�
9categorical_feature_preprocess/string_to_index/hash_table	HashTable*
	container *
value_dtype0	*
use_node_name_sharing( *
shared_name *
_output_shapes
:*
	key_dtype0
�
?categorical_feature_preprocess/string_to_index/hash_table/ConstConst*
dtype0	*
_output_shapes
: *
valueB	 R
���������
�
Dcategorical_feature_preprocess/string_to_index/hash_table/table_initInitializeTable9categorical_feature_preprocess/string_to_index/hash_table4categorical_feature_preprocess/string_to_index/Const3categorical_feature_preprocess/string_to_index/Cast*L
_classB
@>loc:@categorical_feature_preprocess/string_to_index/hash_table*

Tval0	*

Tkey0
�
0categorical_feature_preprocess/hash_table_LookupLookupTableFind9categorical_feature_preprocess/string_to_index/hash_tableExpandDims_5?categorical_feature_preprocess/string_to_index/hash_table/Const*

Tout0	*
_output_shapes

:
*	
Tin0*L
_classB
@>loc:@categorical_feature_preprocess/string_to_index/hash_table
�
6categorical_feature_preprocess/string_to_index_1/ConstConst*
_output_shapes
:*
dtype0*3
value*B(BabcBjklBpqrBmnoBghiBdefB 
w
5categorical_feature_preprocess/string_to_index_1/SizeConst*
_output_shapes
: *
dtype0*
value	B :
~
<categorical_feature_preprocess/string_to_index_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
~
<categorical_feature_preprocess/string_to_index_1/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
6categorical_feature_preprocess/string_to_index_1/rangeRange<categorical_feature_preprocess/string_to_index_1/range/start5categorical_feature_preprocess/string_to_index_1/Size<categorical_feature_preprocess/string_to_index_1/range/delta*
_output_shapes
:*

Tidx0
�
5categorical_feature_preprocess/string_to_index_1/CastCast6categorical_feature_preprocess/string_to_index_1/range*
_output_shapes
:*

DstT0	*

SrcT0
�
;categorical_feature_preprocess/string_to_index_1/hash_table	HashTable*
shared_name *
	key_dtype0*
_output_shapes
:*
use_node_name_sharing( *
value_dtype0	*
	container 
�
Acategorical_feature_preprocess/string_to_index_1/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������
�
Fcategorical_feature_preprocess/string_to_index_1/hash_table/table_initInitializeTable;categorical_feature_preprocess/string_to_index_1/hash_table6categorical_feature_preprocess/string_to_index_1/Const5categorical_feature_preprocess/string_to_index_1/Cast*N
_classD
B@loc:@categorical_feature_preprocess/string_to_index_1/hash_table*

Tval0	*

Tkey0
�
2categorical_feature_preprocess/hash_table_Lookup_1LookupTableFind;categorical_feature_preprocess/string_to_index_1/hash_tableExpandDims_6Acategorical_feature_preprocess/string_to_index_1/hash_table/Const*

Tout0	*
_output_shapes

:
*	
Tin0*N
_classD
B@loc:@categorical_feature_preprocess/string_to_index_1/hash_table
�
6categorical_feature_preprocess/string_to_index_2/ConstConst*
_output_shapes
:*
dtype0*:
value1B/BvanBcarBtrainBdroneBbikeBtruckB 
w
5categorical_feature_preprocess/string_to_index_2/SizeConst*
dtype0*
_output_shapes
: *
value	B :
~
<categorical_feature_preprocess/string_to_index_2/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
~
<categorical_feature_preprocess/string_to_index_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
6categorical_feature_preprocess/string_to_index_2/rangeRange<categorical_feature_preprocess/string_to_index_2/range/start5categorical_feature_preprocess/string_to_index_2/Size<categorical_feature_preprocess/string_to_index_2/range/delta*
_output_shapes
:*

Tidx0
�
5categorical_feature_preprocess/string_to_index_2/CastCast6categorical_feature_preprocess/string_to_index_2/range*
_output_shapes
:*

DstT0	*

SrcT0
�
;categorical_feature_preprocess/string_to_index_2/hash_table	HashTable*
_output_shapes
:*
value_dtype0	*
	container *
	key_dtype0*
use_node_name_sharing( *
shared_name 
�
Acategorical_feature_preprocess/string_to_index_2/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������
�
Fcategorical_feature_preprocess/string_to_index_2/hash_table/table_initInitializeTable;categorical_feature_preprocess/string_to_index_2/hash_table6categorical_feature_preprocess/string_to_index_2/Const5categorical_feature_preprocess/string_to_index_2/Cast*N
_classD
B@loc:@categorical_feature_preprocess/string_to_index_2/hash_table*

Tkey0*

Tval0	
�
2categorical_feature_preprocess/hash_table_Lookup_2LookupTableFind;categorical_feature_preprocess/string_to_index_2/hash_tableExpandDims_7Acategorical_feature_preprocess/string_to_index_2/hash_table/Const*

Tout0	*
_output_shapes

:
*	
Tin0*N
_classD
B@loc:@categorical_feature_preprocess/string_to_index_2/hash_table
�
bdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
�
adnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/CastCastbdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Shape*

SrcT0*
_output_shapes
:*

DstT0	
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast_1/xConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast_1Castednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast_1/x*
_output_shapes
: *

DstT0	*

SrcT0
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/NotEqualNotEqual2categorical_feature_preprocess/hash_table_Lookup_1cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast_1*
_output_shapes

:
*
T0	
�
bdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/WhereWhereednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/NotEqual*'
_output_shapes
:���������
�
jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/ReshapeReshape2categorical_feature_preprocess/hash_table_Lookup_1jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Reshape/shape*
T0	*
_output_shapes
:
*
Tshape0
�
pdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"       
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stackrdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stack_1rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stack_2*
end_mask*
ellipsis_mask *

begin_mask*
shrink_axis_mask*#
_output_shapes
:���������*
new_axis_mask *
T0	*
Index0
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
�
tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
�
tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1StridedSlicebdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Whererdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stacktdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stack_2*'
_output_shapes
:���������*
end_mask*
new_axis_mask *
ellipsis_mask *

begin_mask*
shrink_axis_mask *
T0	*
Index0
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/unstackUnpackadnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
�
bdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/stackPackfdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
�
`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/MulMulldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1bdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/stack*'
_output_shapes
:���������*
T0	
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
�
`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/SumSum`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Mulrdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Sum/reduction_indices*#
_output_shapes
:���������*
T0	*
	keep_dims( *

Tidx0
�
`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/AddAddjdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Sum*#
_output_shapes
:���������*
T0	
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/GatherGatherddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Reshape`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Add*#
_output_shapes
:���������*
validate_indices(*
Tparams0	*
Tindices0	
�
Ndnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/mod/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
�
Ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/modFloorModcdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/GatherNdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/mod/y*#
_output_shapes
:���������*
T0	
�
idnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_sliceStridedSliceadnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Castidnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stackkdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stack_1kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stack_2*
end_mask *
ellipsis_mask *

begin_mask*
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
T0	*
Index0
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
�
mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1StridedSliceadnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Castkdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stackmdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stack_1mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0	*
Index0*
_output_shapes
:*
shrink_axis_mask 
�
[dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
Zdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/ProdProdednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1[dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat/values_1PackZdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/Prod*
N*
T0	*
_output_shapes
:*

axis 
�
adnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
\dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concatConcatV2cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_sliceednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat/values_1adnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat/axis*
_output_shapes
:*
N*
T0	*

Tidx0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshapeSparseReshapebdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Whereadnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast\dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat*-
_output_shapes
:���������:
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape/IdentityIdentityLdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/mod*
T0	*#
_output_shapes
:���������
�
_dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
valueB"      
�
^dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
valueB
 *    
�
`dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
valueB
 *���>
�
idnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal_dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/shape*
seed2 *
dtype0*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*

seed *
_output_shapes

:*
T0
�
]dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/mulMulidnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormal`dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/stddev*
T0*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
Ydnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normalAdd]dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/mul^dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/mean*
T0*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
<dnn/input_from_feature_columns/str2_embedding/weights/part_0
VariableV2*
_output_shapes

:*
dtype0*
shape
:*
	container *O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
shared_name 
�
Cdnn/input_from_feature_columns/str2_embedding/weights/part_0/AssignAssign<dnn/input_from_feature_columns/str2_embedding/weights/part_0Ydnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
Adnn/input_from_feature_columns/str2_embedding/weights/part_0/readIdentity<dnn/input_from_feature_columns/str2_embedding/weights/part_0*
T0*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
�
idnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SliceSliceednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape:1jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice/beginidnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/ProdProdddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Sliceddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
�
mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather/indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GatherGatherednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape:1mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather/indices*
_output_shapes
: *
validate_indices(*
Tparams0	*
Tindices0
�
vdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/new_shapePackcdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Prodednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather*
_output_shapes
:*
N*

axis *
T0	
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshapeSparseReshapecdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshapeednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape:1vdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:���������:
�
udnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/IdentityIdentityldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape/Identity*#
_output_shapes
:���������*
T0	
�
mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GreaterEqualGreaterEqualudnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/Identitymdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GreaterEqual/y*
T0	*#
_output_shapes
:���������
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/WhereWherekdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GreaterEqual*'
_output_shapes
:���������
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/ReshapeReshapeddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Whereldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape/shape*#
_output_shapes
:���������*
Tshape0*
T0	
�
gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_1Gatherldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshapefdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*'
_output_shapes
:���������
�
gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_2Gatherudnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/Identityfdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape*#
_output_shapes
:���������*
validate_indices(*
Tparams0	*
Tindices0	
�
gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/IdentityIdentityndnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
�
xdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlicegdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Identity�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*
_output_shapes
: *
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
Index0*
T0	
�
wdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/CastCast�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice*
_output_shapes
: *

DstT0*

SrcT0	
�
~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
�
~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
xdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/rangeRange~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range/startwdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Cast~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:���������
�
ydnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Cast_1Castxdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range*

SrcT0*#
_output_shapes
:���������*

DstT0	
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlicegdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0	*
Index0*#
_output_shapes
:���������*
shrink_axis_mask
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ListDiffListDiffydnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Cast_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1*2
_output_shapes 
:���������:���������*
out_idx0*
T0	
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlicegdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Identity�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
end_mask *
Index0*
T0	*
shrink_axis_mask*
new_axis_mask 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDims�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*
_output_shapes
:*
T0	*

Tdim0
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
_output_shapes
: *
value	B
 Z
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDense{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ListDiff}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ExpandDims�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_values�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*
Tindices0	*
validate_indices(*
T0
*#
_output_shapes
:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
zdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ReshapeReshape{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ListDiff�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Reshape/shape*
T0	*'
_output_shapes
:���������*
Tshape0
�
}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Reshape*'
_output_shapes
:���������*
T0	
�
~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
�
ydnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concatConcatV2zdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Reshape}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/zeros_like~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat/axis*'
_output_shapes
:���������*
N*
T0	*

Tidx0
�
xdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ShapeShape{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ListDiff*
T0	*
_output_shapes
:*
out_type0
�
wdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/FillFillxdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Shapexdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_1ydnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*
T0	*'
_output_shapes
:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_2wdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Fill�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:���������*
N*
T0	*

Tidx0
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorder{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_1{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_2gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Identity*
T0	*6
_output_shapes$
":���������:���������
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/IdentityIdentitygdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Identity*
_output_shapes
:*
T0	
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSlice�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseReorder�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0	*#
_output_shapes
:���������*
shrink_axis_mask
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/CastCast�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:���������*

DstT0
�
}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/UniqueUnique�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseReorder:1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherAdnn/input_from_feature_columns/str2_embedding/weights/part_0/read}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/Unique*'
_output_shapes
:���������*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
Tparams0*
validate_indices(*
Tindices0	
�
vdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparseSparseSegmentMean�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/embedding_lookupdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/Unique:1{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/Cast*'
_output_shapes
:���������*
T0*

Tidx0
�
ndnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   
�
hdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_1Reshape�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDensendnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_1/shape*
T0
*'
_output_shapes
:���������*
Tshape0
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/ShapeShapevdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse*
_output_shapes
:*
out_type0*
T0
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
�
tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_sliceStridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Shaperdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stacktdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/stackPackfdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/stack/0ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/TileTilehdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_1ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/stack*

Tmultiples0*
T0
*0
_output_shapes
:������������������
�
idnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/zeros_like	ZerosLikevdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse*'
_output_shapes
:���������*
T0
�
^dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweightsSelectcdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Tileidnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/zeros_likevdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/CastCastednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape:1*
_output_shapes
:*

DstT0*

SrcT0	
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1Slicecdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Castldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1/beginkdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Shape_1Shape^dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights*
T0*
_output_shapes
:*
out_type0
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2Slicefdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Shape_1ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2/beginkdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2/size*
_output_shapes
:*
Index0*
T0
�
jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/concatConcatV2fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_2Reshape^dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweightsednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/concat*'
_output_shapes
:���������*
Tshape0*
T0
�
`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
      
�
_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/CastCast`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Shape*
_output_shapes
:*

DstT0	*

SrcT0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast_1Castcdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast_1/x*
_output_shapes
: *

DstT0	*

SrcT0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/NotEqualNotEqual0categorical_feature_preprocess/hash_table_Lookupadnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast_1*
T0	*
_output_shapes

:

�
`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/WhereWherecdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/NotEqual*'
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
bdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/ReshapeReshape0categorical_feature_preprocess/hash_table_Lookuphdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Reshape/shape*
T0	*
_output_shapes
:
*
Tshape0
�
ndnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"       
�
pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
�
pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
hdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Wherendnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stackpdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stack_1pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stack_2*
end_mask*

begin_mask*
ellipsis_mask *
shrink_axis_mask*#
_output_shapes
:���������*
new_axis_mask *
Index0*
T0	
�
pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
�
rdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
�
rdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1StridedSlice`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stackrdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stack_1rdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stack_2*
end_mask*

begin_mask*
ellipsis_mask *
shrink_axis_mask *'
_output_shapes
:���������*
new_axis_mask *
Index0*
T0	
�
bdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/unstackUnpack_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast*
_output_shapes
: : *

axis *	
num*
T0	
�
`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/stackPackddnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
�
^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/MulMuljdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/stack*'
_output_shapes
:���������*
T0	
�
pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
�
^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/SumSum^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Mulpdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Sum/reduction_indices*#
_output_shapes
:���������*
T0	*
	keep_dims( *

Tidx0
�
^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/AddAddhdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Sum*#
_output_shapes
:���������*
T0	
�
adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/GatherGatherbdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Reshape^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Add*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:���������
�
Ldnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/mod/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
�
Jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/modFloorModadnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/GatherLdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/mod/y*
T0	*#
_output_shapes
:���������
�
gdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
�
idnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
idnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Castgdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stackidnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stack_1idnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stack_2*
end_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0	
�
idnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
�
kdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
cdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1StridedSlice_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Castidnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stackkdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stack_1kdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
:*
end_mask*
Index0*
T0	*
shrink_axis_mask *
new_axis_mask 
�
Ydnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Xdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/ProdProdcdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1Ydnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat/values_1PackXdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/Prod*
N*
T0	*
_output_shapes
:*

axis 
�
_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Zdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concatConcatV2adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slicecdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat/values_1_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat/axis*
N*

Tidx0*
T0	*
_output_shapes
:
�
adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshapeSparseReshape`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Where_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/CastZdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat*-
_output_shapes
:���������:
�
jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshape/IdentityIdentityJdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/mod*#
_output_shapes
:���������*
T0	
�
bdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SparseToDenseSparseToDenseadnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshapecdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshape:1jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshape/Identitybdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SparseToDense/default_value*0
_output_shapes
:������������������*
validate_indices(*
T0	*
Tindices0	
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Vdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
�
Wdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/on_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Xdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
Ndnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hotOneHotTdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SparseToDenseTdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/depthWdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/on_valueXdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/off_value*
T0*4
_output_shapes"
 :������������������*
TI0	*
axis���������
�
\dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
Jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SumSumNdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot\dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*'
_output_shapes
:���������
�
`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
�
_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/CastCast`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Shape*
_output_shapes
:*

DstT0	*

SrcT0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast_1Castcdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0	
�
cdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/NotEqualNotEqual2categorical_feature_preprocess/hash_table_Lookup_2adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast_1*
T0	*
_output_shapes

:

�
`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/WhereWherecdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/NotEqual*'
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
�
bdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/ReshapeReshape2categorical_feature_preprocess/hash_table_Lookup_2hdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Reshape/shape*
_output_shapes
:
*
Tshape0*
T0	
�
ndnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"       
�
pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
�
pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
hdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Wherendnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stackpdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stack_1pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stack_2*
end_mask*

begin_mask*
ellipsis_mask *
shrink_axis_mask*#
_output_shapes
:���������*
new_axis_mask *
Index0*
T0	
�
pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
�
rdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
�
rdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1StridedSlice`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stackrdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stack_1rdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *'
_output_shapes
:���������*
end_mask*
Index0*
T0	*
shrink_axis_mask *
new_axis_mask 
�
bdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/unstackUnpack_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
�
`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/stackPackddnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/unstack:1*
_output_shapes
:*
N*

axis *
T0	
�
^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/MulMuljdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/stack*
T0	*'
_output_shapes
:���������
�
pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
�
^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/SumSum^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Mulpdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Sum/reduction_indices*#
_output_shapes
:���������*
T0	*
	keep_dims( *

Tidx0
�
^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/AddAddhdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Sum*#
_output_shapes
:���������*
T0	
�
adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/GatherGatherbdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Reshape^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Add*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:���������
�
Ldnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/mod/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
�
Jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/modFloorModadnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/GatherLdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/mod/y*
T0	*#
_output_shapes
:���������
�
gdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
idnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
idnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Castgdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stackidnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stack_1idnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stack_2*
_output_shapes
:*
end_mask *
new_axis_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
Index0*
T0	
�
idnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
�
kdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
cdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1StridedSlice_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Castidnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stackkdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stack_1kdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
:*
end_mask*
Index0*
T0	*
shrink_axis_mask *
new_axis_mask 
�
Ydnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Xdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/ProdProdcdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1Ydnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat/values_1PackXdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/Prod*
_output_shapes
:*
N*

axis *
T0	
�
_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Zdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concatConcatV2adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slicecdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat/values_1_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat/axis*
N*

Tidx0*
T0	*
_output_shapes
:
�
adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshapeSparseReshape`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Where_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/CastZdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat*-
_output_shapes
:���������:
�
jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshape/IdentityIdentityJdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/mod*
T0	*#
_output_shapes
:���������
�
bdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SparseToDense/default_valueConst*
dtype0	*
_output_shapes
: *
valueB	 R
���������
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SparseToDenseSparseToDenseadnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshapecdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshape:1jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshape/Identitybdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SparseToDense/default_value*0
_output_shapes
:������������������*
validate_indices(*
T0	*
Tindices0	
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Vdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/depthConst*
dtype0*
_output_shapes
: *
value	B :
�
Wdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Xdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
Ndnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hotOneHotTdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SparseToDenseTdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/depthWdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/on_valueXdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/off_value*4
_output_shapes"
 :������������������*
TI0	*
axis���������*
T0
�
\dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
Jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SumSumNdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot\dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/Sum/reduction_indices*'
_output_shapes
:���������*
T0*
	keep_dims( *

Tidx0
�
Ednn/input_from_feature_columns/input_from_feature_columns/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
�
@dnn/input_from_feature_columns/input_from_feature_columns/concatConcatV2hdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_2Jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SumJdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/Sum numerical_feature_preprocess/add"numerical_feature_preprocess/add_1ExpandDims_4Ednn/input_from_feature_columns/input_from_feature_columns/concat/axis*
_output_shapes

:
*
N*
T0*

Tidx0
�
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"   
   
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�?�
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�?�>
�
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
seed2 *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*

seed *
_output_shapes

:
*
T0
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0
�
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0
�
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
_output_shapes

:
*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
use_locking(
�
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0
�
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
_output_shapes
:
*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueB
*    
�
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name *
_output_shapes
:
*
shape:

�
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
_output_shapes
:
*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
use_locking(
�
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0
u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
_output_shapes

:
*
T0
�
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
_output_shapes

:

*
transpose_a( *
T0
o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
_output_shapes
:
*
T0
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*
_output_shapes

:

*
data_formatNHWC*
T0
p
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
_output_shapes

:

*
T0
W
zero_fraction/zeroConst*
_output_shapes
: *
dtype0*
valueB
 *    

zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*
_output_shapes

:


g
zero_fraction/CastCastzero_fraction/Equal*
_output_shapes

:

*

DstT0*

SrcT0

d
zero_fraction/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values
�
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
�
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"
   
   
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�7�
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�7?
�
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*
T0*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*
T0*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape
:

*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
_output_shapes

:

*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
use_locking(
�
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0
�
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*
dtype0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueB
*    
�
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape:
*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0
�
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
_output_shapes
:
*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
use_locking(
�
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:


�
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
_output_shapes

:

*
transpose_a( *
T0
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:

�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*
_output_shapes

:

*
data_formatNHWC*
T0
p
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
_output_shapes

:

*
T0
Y
zero_fraction_1/zeroConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
T0*
_output_shapes

:


k
zero_fraction_1/CastCastzero_fraction_1/Equal*
_output_shapes

:

*

DstT0*

SrcT0

f
zero_fraction_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
�
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values
�
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1_activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_1_activation
�
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"
      
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��!�
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��!?
�
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
dtype0*

seed *
T0*
seed2 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0
�
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0
�
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name *
_output_shapes

:
*
shape
:

�
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
_output_shapes

:
*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
use_locking(
�
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
1dnn/hiddenlayer_2/biases/part_0/Initializer/ConstConst*
_output_shapes
:*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB*    
�
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
�
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/Const*
_output_shapes
:*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
use_locking(
�
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
_output_shapes

:
*
T0
�
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *
_output_shapes

:
*
transpose_a( *
T0
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*
data_formatNHWC*
T0*
_output_shapes

:

p
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
_output_shapes

:
*
T0
Y
zero_fraction_2/zeroConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reluzero_fraction_2/zero*
T0*
_output_shapes

:

k
zero_fraction_2/CastCastzero_fraction_2/Equal*

SrcT0
*
_output_shapes

:
*

DstT0
f
zero_fraction_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
�
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.dnn/hiddenlayer_2_fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_2_fraction_of_zero_values
�
)dnn/hiddenlayer_2_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2_fraction_of_zero_values/tagszero_fraction_2/Mean*
_output_shapes
: *
T0
}
 dnn/hiddenlayer_2_activation/tagConst*
dtype0*
_output_shapes
: *-
value$B" Bdnn/hiddenlayer_2_activation
�
dnn/hiddenlayer_2_activationHistogramSummary dnn/hiddenlayer_2_activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
�
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      
�
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *׳]�
�
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *׳]?
�
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0
�
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *,
_class"
 loc:@dnn/logits/weights/part_0
�
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*
T0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0
�
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0*
T0
�
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
�
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0
�
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0*
T0
�
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    
�
dnn/logits/biases/part_0
VariableV2*
	container *
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name *
_output_shapes
:*
shape:
�
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0
�
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0*
T0
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
_output_shapes

:*
T0
�
dnn/logits/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/logits/weights*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a( 
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
_output_shapes
:*
T0
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*
data_formatNHWC*
T0*
_output_shapes

:

Y
zero_fraction_3/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    
q
zero_fraction_3/EqualEqualdnn/logits/BiasAddzero_fraction_3/zero*
_output_shapes

:
*
T0
k
zero_fraction_3/CastCastzero_fraction_3/Equal*
_output_shapes

:
*

DstT0*

SrcT0

f
zero_fraction_3/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
zero_fraction_3/MeanMeanzero_fraction_3/Castzero_fraction_3/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
'dnn/logits_fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*3
value*B( B"dnn/logits_fraction_of_zero_values
�
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_3/Mean*
T0*
_output_shapes
: 
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
a
predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
_output_shapes

:
*
T0
_
predictions/classes/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
�
predictions/classesArgMaxdnn/logits/BiasAddpredictions/classes/dimension*

Tidx0*
T0*
_output_shapes
:

�
0training_loss/softmax_cross_entropy_loss/SqueezeSqueeze+target_feature_preprocess/hash_table_Lookup*
_output_shapes
:
*
T0	*
squeeze_dims

x
.training_loss/softmax_cross_entropy_loss/ShapeConst*
dtype0*
_output_shapes
:*
valueB:

�
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*$
_output_shapes
:
:
*
Tlabels0	*
T0
]
training_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
 training_loss/ScalarSummary/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bloss
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
�
,metrics/remove_squeezable_dimensions/SqueezeSqueeze+target_feature_preprocess/hash_table_Lookup*
T0	*
_output_shapes
:
*
squeeze_dims

���������
~
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*
_output_shapes
:

Z
metrics/ToFloatCastmetrics/Equal*
_output_shapes
:
*

DstT0*

SrcT0

[
metrics/accuracy/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    
z
metrics/accuracy/total
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *)
_class
loc:@metrics/accuracy/total
�
metrics/accuracy/total/readIdentitymetrics/accuracy/total*
T0*
_output_shapes
: *)
_class
loc:@metrics/accuracy/total
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
dtype0*
shared_name *
shape: 
�
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *)
_class
loc:@metrics/accuracy/count
�
metrics/accuracy/count/readIdentitymetrics/accuracy/count*
_output_shapes
: *)
_class
loc:@metrics/accuracy/count*
T0
W
metrics/accuracy/SizeConst*
_output_shapes
: *
dtype0*
value	B :

i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*

SrcT0*
_output_shapes
: *

DstT0
`
metrics/accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*
_output_shapes
: *)
_class
loc:@metrics/accuracy/total*
T0*
use_locking( 
�
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
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
T0*
_output_shapes
: 
]
metrics/accuracy/value/eConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
T0*
_output_shapes
: 
a
metrics/accuracy/Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
_output_shapes
: *
T0
�
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
�
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
_output_shapes
: *
T0
N
metrics/RankConst*
dtype0*
_output_shapes
: *
value	B :
U
metrics/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B :
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
_output_shapes
: *
T0
�
metrics/Assert/ConstConst*
dtype0*
_output_shapes
: *N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]
�
metrics/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*

T
2*
	summarize
�
metrics/Reshape/shapeConst^metrics/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������
�
metrics/ReshapeReshape+target_feature_preprocess/hash_table_Lookupmetrics/Reshape/shape*
_output_shapes
:
*
Tshape0*
T0	
]
metrics/one_hot/on_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
�
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
T0*
_output_shapes

:
*
TI0	*
axis���������
]
metrics/CastCastmetrics/one_hot*

SrcT0*
_output_shapes

:
*

DstT0

j
metrics/auc/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   
�
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*
T0*
_output_shapes

:*
Tshape0
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"   ����
�
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
_output_shapes

:*
Tshape0*
T0

�
metrics/auc/ConstConst*
dtype0*
_output_shapes	
:�*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?
d
metrics/auc/ExpandDims/dimConst*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*
T0*
_output_shapes
:	�*

Tdim0
b
metrics/auc/stackConst*
_output_shapes
:*
dtype0*
valueB"      

metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*
_output_shapes
:	�
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
T0*
_output_shapes
: 
]
metrics/auc/transpose/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
_output_shapes
: *
T0
c
!metrics/auc/transpose/Range/startConst*
_output_shapes
: *
dtype0*
value	B : 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
T0*
_output_shapes

:
m
metrics/auc/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"�      
�
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*
_output_shapes
:	�*
T0*

Tmultiples0
n
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
_output_shapes
:	�*
T0
Z
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*
_output_shapes
:	�
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
_output_shapes
:*
valueB"�      
�
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*
_output_shapes
:	�*
T0
*

Tmultiples0
[
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2*
_output_shapes
:	�
`
metrics/auc/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    
�
metrics/auc/true_positives
VariableV2*
shared_name *
dtype0*
shape:�*
_output_shapes	
:�*
	container 
�
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
_output_shapes	
:�*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
T0*
use_locking(
�
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*
T0*
_output_shapes	
:�*-
_class#
!loc:@metrics/auc/true_positives
n
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater*
_output_shapes
:	�
n
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*
_output_shapes
:	�*

DstT0*

SrcT0

c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*
_output_shapes	
:�*-
_class#
!loc:@metrics/auc/true_positives*
T0*
use_locking( 
b
metrics/auc/zeros_1Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
metrics/auc/false_negatives
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
shape:�
�
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_negatives
�
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_negatives*
T0
s
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot*
_output_shapes
:	�
p
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

SrcT0
*
_output_shapes
:	�*

DstT0
e
#metrics/auc/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
�
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*
use_locking( *
T0*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_negatives
b
metrics/auc/zeros_2Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
metrics/auc/true_negatives
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
shape:�
�
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
_output_shapes	
:�*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
use_locking(
�
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*
T0*
_output_shapes	
:�*-
_class#
!loc:@metrics/auc/true_negatives
y
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot*
_output_shapes
:	�
p
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

SrcT0
*
_output_shapes
:	�*

DstT0
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*
_output_shapes	
:�*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
use_locking( 
b
metrics/auc/zeros_3Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
metrics/auc/false_positives
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
shape:�
�
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
_output_shapes	
:�*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
T0*
use_locking(
�
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*
T0*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_positives
v
metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater*
_output_shapes
:	�
p
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*
_output_shapes
:	�*

DstT0*

SrcT0

e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*
use_locking( *
T0*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_positives
V
metrics/auc/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:�
�
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
_output_shapes	
:�*
T0
X
metrics/auc/add_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
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
dtype0*
_output_shapes
: *
valueB
 *�7�5
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
_output_shapes	
:�*
T0
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
_output_shapes	
:�*
T0
i
metrics/auc/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
l
!metrics/auc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
k
!metrics/auc/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
metrics/auc/strided_sliceStridedSlicemetrics/auc/div_1metrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
_output_shapes	
:�*
end_mask *
T0*
Index0*
shrink_axis_mask *
new_axis_mask 
k
!metrics/auc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
m
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
m
#metrics/auc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes	
:�*
end_mask*
Index0*
T0*
shrink_axis_mask *
new_axis_mask 
t
metrics/auc/subSubmetrics/auc/strided_slicemetrics/auc/strided_slice_1*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
n
#metrics/auc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
m
#metrics/auc/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
_output_shapes	
:�*
end_mask *
new_axis_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
T0*
Index0
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
m
#metrics/auc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
_output_shapes	
:�*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_2metrics/auc/strided_slice_3*
T0*
_output_shapes	
:�
Z
metrics/auc/truediv/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
_output_shapes	
:�*
T0
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
_output_shapes	
:�*
T0
]
metrics/auc/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
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
 *�7�5
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
metrics/auc/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:�
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
_output_shapes	
:�*
T0
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:�
Y
metrics/auc/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5
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
!metrics/auc/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 
n
#metrics/auc/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*
valueB:�
m
#metrics/auc/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*

begin_mask*
ellipsis_mask *
_output_shapes	
:�*
end_mask *
Index0*
T0*
shrink_axis_mask *
new_axis_mask 
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#metrics/auc/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
_output_shapes	
:�*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_4metrics/auc/strided_slice_5*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_6/stackConst*
dtype0*
_output_shapes
:*
valueB: 
n
#metrics/auc/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
m
#metrics/auc/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
end_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
_output_shapes	
:�*
new_axis_mask *
Index0*
T0
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
shrink_axis_mask *
_output_shapes	
:�*
Index0*
T0*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_6metrics/auc/strided_slice_7*
T0*
_output_shapes	
:�
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
:�*
T0
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:�
]
metrics/auc/Const_2Const*
valueB: *
_output_shapes
:*
dtype0
�
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
*metrics/softmax_cross_entropy_loss/SqueezeSqueeze+target_feature_preprocess/hash_table_Lookup*
squeeze_dims
*
_output_shapes
:
*
T0	
r
(metrics/softmax_cross_entropy_loss/ShapeConst*
valueB:
*
_output_shapes
:*
dtype0
�
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*$
_output_shapes
:
:
*
Tlabels0	
a
metrics/eval_loss/ConstConst*
valueB: *
dtype0*
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
metrics/mean/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
v
metrics/mean/total
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*%
_class
loc:@metrics/mean/total*
_output_shapes
: *
T0*
validate_shape(*
use_locking(

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
_output_shapes
: *
T0
Y
metrics/mean/zeros_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
v
metrics/mean/count
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
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
T0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
S
metrics/mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*
_output_shapes
: *

DstT0*

SrcT0
U
metrics/mean/ConstConst*
valueB *
_output_shapes
: *
dtype0
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*
use_locking( *
T0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*
use_locking( *
T0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
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
 *    *
_output_shapes
: *
dtype0

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
_output_shapes
: *
T0
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
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
`

group_depsNoOp^metrics/mean/update_op^metrics/auc/update_op^metrics/accuracy/update_op
\
eval_step/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	eval_step
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
�
eval_step/AssignAssign	eval_stepeval_step/initial_value*
_class
loc:@eval_step*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
_output_shapes
: *
T0
T
AssignAdd/valueConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
_output_shapes
: *
T0*
use_locking( 
�
initNoOp^global_step/AssignD^dnn/input_from_feature_columns/str2_embedding/weights/part_0/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign(^dnn/hiddenlayer_2/weights/part_0/Assign'^dnn/hiddenlayer_2/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
_output_shapes
: *
dtype0	
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized<dnn/input_from_feature_columns/str2_embedding/weights/part_0*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitialized"input_producer/limit_epochs/epochs*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
_output_shapes
: *
dtype0	
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedmetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedmetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedmetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedmetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedmetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedmetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedmetrics/mean/total*%
_class
loc:@metrics/mean/total*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedmetrics/mean/count*%
_class
loc:@metrics/mean/count*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized	eval_step*
_class
loc:@eval_step*
_output_shapes
: *
dtype0
�	
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_19*
T0
*

axis *
N*
_output_shapes
:
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst*�
value�B�Bglobal_stepB<dnn/input_from_feature_columns/str2_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B"input_producer/limit_epochs/epochsBmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/countB	eval_step*
_output_shapes
:*
dtype0
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
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
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
shrink_axis_mask *
_output_shapes
: *
Index0*
T0*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask 
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
T0*

axis *
N*
_output_shapes
:
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
_output_shapes
:*
T0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
Tshape0*
_output_shapes
:
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
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:���������
g
$report_uninitialized_resources/ConstConst*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized<dnn/input_from_feature_columns/str2_embedding/weights/part_0*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: *
dtype0
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: *
dtype0
�
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializeddnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: *
dtype0
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_9*

axis *
_output_shapes
:
*
T0
*
N

}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:

�
&report_uninitialized_variables_1/ConstConst*�
value�B�
Bglobal_stepB<dnn/input_from_feature_columns/str2_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0*
_output_shapes
:
*
dtype0
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
shrink_axis_mask *
_output_shapes
:*
Index0*
T0*
end_mask *
new_axis_mask *

begin_mask*
ellipsis_mask 
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
shrink_axis_mask *
Index0*
T0*
end_mask*
_output_shapes
: *

begin_mask *
ellipsis_mask 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
T0*

axis *
N*
_output_shapes
:
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
_output_shapes
:
*
T0
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
_output_shapes
:
*
T0

�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*#
_output_shapes
:���������*
T0	
�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*#
_output_shapes
:���������*
validate_indices(*
Tparams0*
Tindices0	
�
init_2NoOp*^input_producer/limit_epochs/epochs/Assign^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign^eval_step/Assign
�
init_all_tablesNoOp@^target_feature_preprocess/string_to_index/hash_table/table_initE^categorical_feature_preprocess/string_to_index/hash_table/table_initG^categorical_feature_preprocess/string_to_index_1/hash_table/table_initG^categorical_feature_preprocess/string_to_index_2/hash_table/table_init
/
group_deps_2NoOp^init_2^init_all_tables
�
Merge/MergeSummaryMergeSummary"input_producer/fraction_of_32_fullbatch/fraction_of_150_full)dnn/hiddenlayer_0_fraction_of_zero_valuesdnn/hiddenlayer_0_activation)dnn/hiddenlayer_1_fraction_of_zero_valuesdnn/hiddenlayer_1_activation)dnn/hiddenlayer_2_fraction_of_zero_valuesdnn/hiddenlayer_2_activation"dnn/logits_fraction_of_zero_valuesdnn/logits_activationtraining_loss/ScalarSummary*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_691d66c29d724414a6f917008f91879d/part*
dtype0*
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
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�
Bdnn/hiddenlayer_0/biasesBdnn/hiddenlayer_0/weightsBdnn/hiddenlayer_1/biasesBdnn/hiddenlayer_1/weightsBdnn/hiddenlayer_2/biasesBdnn/hiddenlayer_2/weightsB5dnn/input_from_feature_columns/str2_embedding/weightsBdnn/logits/biasesBdnn/logits/weightsBglobal_step*
_output_shapes
:
*
dtype0
�
save/SaveV2/shape_and_slicesConst*�
valuewBu
B10 0,10B21 10 0,21:0,10B10 0,10B10 10 0,10:0,10B5 0,5B10 5 0,10:0,5B7 3 0,7:0,3B3 0,3B5 3 0,5:0,3B *
dtype0*
_output_shapes
:

�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read%dnn/hiddenlayer_0/weights/part_0/read$dnn/hiddenlayer_1/biases/part_0/read%dnn/hiddenlayer_1/weights/part_0/read$dnn/hiddenlayer_2/biases/part_0/read%dnn/hiddenlayer_2/weights/part_0/readAdnn/input_from_feature_columns/str2_embedding/weights/part_0/readdnn/logits/biases/part_0/readdnn/logits/weights/part_0/readglobal_step*
dtypes
2
	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
�
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
valueBB10 0,10*
_output_shapes
:*
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssigndnn/hiddenlayer_0/biases/part_0save/RestoreV2*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
validate_shape(*
_output_shapes
:


save/RestoreV2_1/tensor_namesConst*.
value%B#Bdnn/hiddenlayer_0/weights*
dtype0*
_output_shapes
:
y
!save/RestoreV2_1/shape_and_slicesConst*$
valueBB21 10 0,21:0,10*
_output_shapes
:*
dtype0
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assign dnn/hiddenlayer_0/weights/part_0save/RestoreV2_1*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
~
save/RestoreV2_2/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_1/biases*
dtype0*
_output_shapes
:
q
!save/RestoreV2_2/shape_and_slicesConst*
valueBB10 0,10*
_output_shapes
:*
dtype0
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assigndnn/hiddenlayer_1/biases/part_0save/RestoreV2_2*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(

save/RestoreV2_3/tensor_namesConst*.
value%B#Bdnn/hiddenlayer_1/weights*
dtype0*
_output_shapes
:
y
!save/RestoreV2_3/shape_and_slicesConst*$
valueBB10 10 0,10:0,10*
dtype0*
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
save/Assign_3Assign dnn/hiddenlayer_1/weights/part_0save/RestoreV2_3*
use_locking(*
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
validate_shape(*
_output_shapes

:


~
save/RestoreV2_4/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_2/biases*
_output_shapes
:*
dtype0
o
!save/RestoreV2_4/shape_and_slicesConst*
valueBB5 0,5*
_output_shapes
:*
dtype0
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
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
value%B#Bdnn/hiddenlayer_2/weights*
dtype0*
_output_shapes
:
w
!save/RestoreV2_5/shape_and_slicesConst*"
valueBB10 5 0,10:0,5*
_output_shapes
:*
dtype0
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assign dnn/hiddenlayer_2/weights/part_0save/RestoreV2_5*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_6/tensor_namesConst*J
valueAB?B5dnn/input_from_feature_columns/str2_embedding/weights*
_output_shapes
:*
dtype0
u
!save/RestoreV2_6/shape_and_slicesConst* 
valueBB7 3 0,7:0,3*
_output_shapes
:*
dtype0
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assign<dnn/input_from_feature_columns/str2_embedding/weights/part_0save/RestoreV2_6*
use_locking(*
T0*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
validate_shape(*
_output_shapes

:
w
save/RestoreV2_7/tensor_namesConst*&
valueBBdnn/logits/biases*
dtype0*
_output_shapes
:
o
!save/RestoreV2_7/shape_and_slicesConst*
valueBB3 0,3*
_output_shapes
:*
dtype0
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7Assigndnn/logits/biases/part_0save/RestoreV2_7*
use_locking(*
T0*+
_class!
loc:@dnn/logits/biases/part_0*
validate_shape(*
_output_shapes
:
x
save/RestoreV2_8/tensor_namesConst*'
valueBBdnn/logits/weights*
dtype0*
_output_shapes
:
u
!save/RestoreV2_8/shape_and_slicesConst* 
valueBB5 3 0,5:0,3*
_output_shapes
:*
dtype0
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assigndnn/logits/weights/part_0save/RestoreV2_8*
use_locking(*
T0*,
_class"
 loc:@dnn/logits/weights/part_0*
validate_shape(*
_output_shapes

:
q
save/RestoreV2_9/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2	
�
save/Assign_9Assignglobal_stepsave/RestoreV2_9*
_class
loc:@global_step*
_output_shapes
: *
T0	*
validate_shape(*
use_locking(
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"���uH�     ��	�W�7�AJ��
�9�9
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
ref"T�
output"T"
limitint"
Ttype:
2	
�
	DecodeCSV
records
record_defaults2OUT_TYPE
output2OUT_TYPE"$
OUT_TYPE
list(type)(0:
2	"
field_delimstring,
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
�
	HashTable
table_handle�"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
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
`
InitializeTable
table_handle�
keys"Tkey
values"Tval"
Tkeytype"
Tvaltype
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

u
LookupTableFind
table_handle�
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
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
D
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
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
z
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint���������
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
2	�
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
z
TextLineReaderV2
reader_handle"
skip_header_linesint "
	containerstring "
shared_namestring �
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
2	�
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
Ttype*1.0.12v1.0.0-65-g4763edf-dirty�


global_step/Initializer/ConstConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
global_step
VariableV2*
shared_name *
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
�
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
T0	*
_class
loc:@global_step*
_output_shapes
: 
}
input_producer/ConstConst*5
value,B*B /tmp/tmp8TBLUm/eval_csv_data.csv*
_output_shapes
:*
dtype0
U
input_producer/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Z
input_producer/Greater/yConst*
value	B : *
_output_shapes
: *
dtype0
q
input_producer/GreaterGreaterinput_producer/Sizeinput_producer/Greater/y*
_output_shapes
: *
T0
�
input_producer/Assert/ConstConst*G
value>B< B6string_input_producer requires a non-null input tensor*
dtype0*
_output_shapes
: 
�
#input_producer/Assert/Assert/data_0Const*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: *
dtype0
�
input_producer/Assert/AssertAssertinput_producer/Greater#input_producer/Assert/Assert/data_0*

T
2*
	summarize
}
input_producer/IdentityIdentityinput_producer/Const^input_producer/Assert/Assert*
_output_shapes
:*
T0
c
!input_producer/limit_epochs/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
"input_producer/limit_epochs/epochs
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
�
)input_producer/limit_epochs/epochs/AssignAssign"input_producer/limit_epochs/epochs!input_producer/limit_epochs/Const*
use_locking(*
T0	*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
validate_shape(*
_output_shapes
: 
�
'input_producer/limit_epochs/epochs/readIdentity"input_producer/limit_epochs/epochs*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
_output_shapes
: *
T0	
�
%input_producer/limit_epochs/CountUpTo	CountUpTo"input_producer/limit_epochs/epochs*
T0	*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
_output_shapes
: *
limit
�
input_producer/limit_epochsIdentityinput_producer/Identity&^input_producer/limit_epochs/CountUpTo*
T0*
_output_shapes
:
�
input_producerFIFOQueueV2*
shapes
: *
	container *
_output_shapes
: *
component_types
2*
capacity *
shared_name 
�
)input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producerinput_producer/limit_epochs*
Tcomponents
2*

timeout_ms���������
b
#input_producer/input_producer_CloseQueueCloseV2input_producer*
cancel_pending_enqueues( 
d
%input_producer/input_producer_Close_1QueueCloseV2input_producer*
cancel_pending_enqueues(
Y
"input_producer/input_producer_SizeQueueSizeV2input_producer*
_output_shapes
: 
o
input_producer/CastCast"input_producer/input_producer_Size*

SrcT0*
_output_shapes
: *

DstT0
Y
input_producer/mul/yConst*
valueB
 *   =*
_output_shapes
: *
dtype0
e
input_producer/mulMulinput_producer/Castinput_producer/mul/y*
T0*
_output_shapes
: 
�
'input_producer/fraction_of_32_full/tagsConst*3
value*B( B"input_producer/fraction_of_32_full*
_output_shapes
: *
dtype0
�
"input_producer/fraction_of_32_fullScalarSummary'input_producer/fraction_of_32_full/tagsinput_producer/mul*
_output_shapes
: *
T0
y
TextLineReaderV2TextLineReaderV2*
skip_header_lines *
shared_name *
_output_shapes
: *
	container 
^
ReaderReadUpToV2/num_recordsConst*
value	B	 R
*
_output_shapes
: *
dtype0	
�
ReaderReadUpToV2ReaderReadUpToV2TextLineReaderV2input_producerReaderReadUpToV2/num_records*2
_output_shapes 
:���������:���������
M
batch/ConstConst*
value	B
 Z*
_output_shapes
: *
dtype0

�
batch/fifo_queueFIFOQueueV2*
shapes
: : *
	container *
_output_shapes
: *
component_types
2*
capacity�*
shared_name 
X
batch/cond/SwitchSwitchbatch/Constbatch/Const*
T0
*
_output_shapes
: : 
U
batch/cond/switch_tIdentitybatch/cond/Switch:1*
T0
*
_output_shapes
: 
S
batch/cond/switch_fIdentitybatch/cond/Switch*
_output_shapes
: *
T0

L
batch/cond/pred_idIdentitybatch/Const*
T0
*
_output_shapes
: 
�
(batch/cond/fifo_queue_EnqueueMany/SwitchSwitchbatch/fifo_queuebatch/cond/pred_id*#
_class
loc:@batch/fifo_queue*
_output_shapes
: : *
T0
�
*batch/cond/fifo_queue_EnqueueMany/Switch_1SwitchReaderReadUpToV2batch/cond/pred_id*
T0*#
_class
loc:@ReaderReadUpToV2*2
_output_shapes 
:���������:���������
�
*batch/cond/fifo_queue_EnqueueMany/Switch_2SwitchReaderReadUpToV2:1batch/cond/pred_id*#
_class
loc:@ReaderReadUpToV2*2
_output_shapes 
:���������:���������*
T0
�
!batch/cond/fifo_queue_EnqueueManyQueueEnqueueManyV2*batch/cond/fifo_queue_EnqueueMany/Switch:1,batch/cond/fifo_queue_EnqueueMany/Switch_1:1,batch/cond/fifo_queue_EnqueueMany/Switch_2:1*
Tcomponents
2*

timeout_ms���������
�
batch/cond/control_dependencyIdentitybatch/cond/switch_t"^batch/cond/fifo_queue_EnqueueMany*&
_class
loc:@batch/cond/switch_t*
_output_shapes
: *
T0

-
batch/cond/NoOpNoOp^batch/cond/switch_f
�
batch/cond/control_dependency_1Identitybatch/cond/switch_f^batch/cond/NoOp*
T0
*&
_class
loc:@batch/cond/switch_f*
_output_shapes
: 
�
batch/cond/MergeMergebatch/cond/control_dependency_1batch/cond/control_dependency*
_output_shapes
: : *
T0
*
N
W
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue*
cancel_pending_enqueues( 
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*

SrcT0*
_output_shapes
: *

DstT0
P
batch/mul/yConst*
valueB
 *t�;*
_output_shapes
: *
dtype0
J
	batch/mulMul
batch/Castbatch/mul/y*
_output_shapes
: *
T0
z
batch/fraction_of_150_full/tagsConst*+
value"B  Bbatch/fraction_of_150_full*
_output_shapes
: *
dtype0
x
batch/fraction_of_150_fullScalarSummarybatch/fraction_of_150_full/tags	batch/mul*
T0*
_output_shapes
: 
I
batch/nConst*
value	B :
*
dtype0*
_output_shapes
: 
�
batchQueueDequeueManyV2batch/fifo_queuebatch/n*

timeout_ms���������* 
_output_shapes
:
:
*
component_types
2
i
 csv_to_tensors/record_defaults_0Const*
valueB
B *
_output_shapes
:*
dtype0
i
 csv_to_tensors/record_defaults_1Const*
valueB
B *
dtype0*
_output_shapes
:
m
 csv_to_tensors/record_defaults_2Const*
valueB*��sA*
_output_shapes
:*
dtype0
m
 csv_to_tensors/record_defaults_3Const*
valueB*�pA*
_output_shapes
:*
dtype0
m
 csv_to_tensors/record_defaults_4Const*
valueB*ײ�@*
_output_shapes
:*
dtype0
i
 csv_to_tensors/record_defaults_5Const*
valueB
B *
_output_shapes
:*
dtype0
i
 csv_to_tensors/record_defaults_6Const*
valueB
B *
dtype0*
_output_shapes
:
i
 csv_to_tensors/record_defaults_7Const*
valueB
B *
dtype0*
_output_shapes
:
�
csv_to_tensors	DecodeCSVbatch:1 csv_to_tensors/record_defaults_0 csv_to_tensors/record_defaults_1 csv_to_tensors/record_defaults_2 csv_to_tensors/record_defaults_3 csv_to_tensors/record_defaults_4 csv_to_tensors/record_defaults_5 csv_to_tensors/record_defaults_6 csv_to_tensors/record_defaults_7*
OUT_TYPE

2*D
_output_shapes2
0:
:
:
:
:
:
:
:
*
field_delim,
P
ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
m

ExpandDims
ExpandDimscsv_to_tensorsExpandDims/dim*

Tdim0*
T0*
_output_shapes

:

R
ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
s
ExpandDims_1
ExpandDimscsv_to_tensors:1ExpandDims_1/dim*

Tdim0*
_output_shapes

:
*
T0
R
ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 
s
ExpandDims_2
ExpandDimscsv_to_tensors:2ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes

:

R
ExpandDims_3/dimConst*
value	B :*
_output_shapes
: *
dtype0
s
ExpandDims_3
ExpandDimscsv_to_tensors:3ExpandDims_3/dim*

Tdim0*
_output_shapes

:
*
T0
R
ExpandDims_4/dimConst*
value	B :*
_output_shapes
: *
dtype0
s
ExpandDims_4
ExpandDimscsv_to_tensors:4ExpandDims_4/dim*

Tdim0*
_output_shapes

:
*
T0
R
ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: 
s
ExpandDims_5
ExpandDimscsv_to_tensors:5ExpandDims_5/dim*

Tdim0*
_output_shapes

:
*
T0
R
ExpandDims_6/dimConst*
value	B :*
_output_shapes
: *
dtype0
s
ExpandDims_6
ExpandDimscsv_to_tensors:6ExpandDims_6/dim*

Tdim0*
_output_shapes

:
*
T0
R
ExpandDims_7/dimConst*
value	B :*
_output_shapes
: *
dtype0
s
ExpandDims_7
ExpandDimscsv_to_tensors:7ExpandDims_7/dim*

Tdim0*
T0*
_output_shapes

:

g
"numerical_feature_preprocess/Sub/yConst*
valueB
 *�.<*
_output_shapes
: *
dtype0
�
 numerical_feature_preprocess/SubSubExpandDims_2"numerical_feature_preprocess/Sub/y*
T0*
_output_shapes

:

g
"numerical_feature_preprocess/ConstConst*
valueB
 *   @*
_output_shapes
: *
dtype0
�
 numerical_feature_preprocess/mulMul numerical_feature_preprocess/Sub"numerical_feature_preprocess/Const*
T0*
_output_shapes

:

i
$numerical_feature_preprocess/Const_1Const*
valueB
 *��A*
dtype0*
_output_shapes
: 
�
$numerical_feature_preprocess/truedivRealDiv numerical_feature_preprocess/mul$numerical_feature_preprocess/Const_1*
_output_shapes

:
*
T0
i
$numerical_feature_preprocess/Const_2Const*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
 numerical_feature_preprocess/addAdd$numerical_feature_preprocess/truediv$numerical_feature_preprocess/Const_2*
_output_shapes

:
*
T0
i
$numerical_feature_preprocess/Sub_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"numerical_feature_preprocess/Sub_1SubExpandDims_3$numerical_feature_preprocess/Sub_1/y*
_output_shapes

:
*
T0
i
$numerical_feature_preprocess/Const_3Const*
valueB
 *   A*
dtype0*
_output_shapes
: 
�
"numerical_feature_preprocess/mul_1Mul"numerical_feature_preprocess/Sub_1$numerical_feature_preprocess/Const_3*
T0*
_output_shapes

:

i
$numerical_feature_preprocess/Const_4Const*
valueB
 *  �A*
_output_shapes
: *
dtype0
�
&numerical_feature_preprocess/truediv_1RealDiv"numerical_feature_preprocess/mul_1$numerical_feature_preprocess/Const_4*
T0*
_output_shapes

:

i
$numerical_feature_preprocess/Const_5Const*
valueB
 *  ��*
_output_shapes
: *
dtype0
�
"numerical_feature_preprocess/add_1Add&numerical_feature_preprocess/truediv_1$numerical_feature_preprocess/Const_5*
_output_shapes

:
*
T0
�
/target_feature_preprocess/string_to_index/ConstConst*"
valueBB102B100B101*
dtype0*
_output_shapes
:
p
.target_feature_preprocess/string_to_index/SizeConst*
value	B :*
_output_shapes
: *
dtype0
w
5target_feature_preprocess/string_to_index/range/startConst*
value	B : *
_output_shapes
: *
dtype0
w
5target_feature_preprocess/string_to_index/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
�
/target_feature_preprocess/string_to_index/rangeRange5target_feature_preprocess/string_to_index/range/start.target_feature_preprocess/string_to_index/Size5target_feature_preprocess/string_to_index/range/delta*

Tidx0*
_output_shapes
:
�
.target_feature_preprocess/string_to_index/CastCast/target_feature_preprocess/string_to_index/range*

SrcT0*
_output_shapes
:*

DstT0	
�
4target_feature_preprocess/string_to_index/hash_table	HashTable*
shared_name *
use_node_name_sharing( *
	key_dtype0*
	container *
value_dtype0	*
_output_shapes
:
�
:target_feature_preprocess/string_to_index/hash_table/ConstConst*
valueB	 R
���������*
dtype0	*
_output_shapes
: 
�
?target_feature_preprocess/string_to_index/hash_table/table_initInitializeTable4target_feature_preprocess/string_to_index/hash_table/target_feature_preprocess/string_to_index/Const.target_feature_preprocess/string_to_index/Cast*

Tkey0*

Tval0	*G
_class=
;9loc:@target_feature_preprocess/string_to_index/hash_table
�
+target_feature_preprocess/hash_table_LookupLookupTableFind4target_feature_preprocess/string_to_index/hash_tableExpandDims_1:target_feature_preprocess/string_to_index/hash_table/Const*

Tout0	*G
_class=
;9loc:@target_feature_preprocess/string_to_index/hash_table*
_output_shapes

:
*	
Tin0
�
4categorical_feature_preprocess/string_to_index/ConstConst*C
value:B8BblueBbrownByellowBpinkBblackBgreenBredB *
dtype0*
_output_shapes
:
u
3categorical_feature_preprocess/string_to_index/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
:categorical_feature_preprocess/string_to_index/range/startConst*
value	B : *
_output_shapes
: *
dtype0
|
:categorical_feature_preprocess/string_to_index/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
�
4categorical_feature_preprocess/string_to_index/rangeRange:categorical_feature_preprocess/string_to_index/range/start3categorical_feature_preprocess/string_to_index/Size:categorical_feature_preprocess/string_to_index/range/delta*

Tidx0*
_output_shapes
:
�
3categorical_feature_preprocess/string_to_index/CastCast4categorical_feature_preprocess/string_to_index/range*
_output_shapes
:*

DstT0	*

SrcT0
�
9categorical_feature_preprocess/string_to_index/hash_table	HashTable*
use_node_name_sharing( *
_output_shapes
:*
value_dtype0	*
shared_name *
	container *
	key_dtype0
�
?categorical_feature_preprocess/string_to_index/hash_table/ConstConst*
valueB	 R
���������*
_output_shapes
: *
dtype0	
�
Dcategorical_feature_preprocess/string_to_index/hash_table/table_initInitializeTable9categorical_feature_preprocess/string_to_index/hash_table4categorical_feature_preprocess/string_to_index/Const3categorical_feature_preprocess/string_to_index/Cast*L
_classB
@>loc:@categorical_feature_preprocess/string_to_index/hash_table*

Tval0	*

Tkey0
�
0categorical_feature_preprocess/hash_table_LookupLookupTableFind9categorical_feature_preprocess/string_to_index/hash_tableExpandDims_5?categorical_feature_preprocess/string_to_index/hash_table/Const*

Tout0	*L
_classB
@>loc:@categorical_feature_preprocess/string_to_index/hash_table*
_output_shapes

:
*	
Tin0
�
6categorical_feature_preprocess/string_to_index_1/ConstConst*3
value*B(BabcBjklBpqrBmnoBghiBdefB *
dtype0*
_output_shapes
:
w
5categorical_feature_preprocess/string_to_index_1/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
~
<categorical_feature_preprocess/string_to_index_1/range/startConst*
value	B : *
_output_shapes
: *
dtype0
~
<categorical_feature_preprocess/string_to_index_1/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
�
6categorical_feature_preprocess/string_to_index_1/rangeRange<categorical_feature_preprocess/string_to_index_1/range/start5categorical_feature_preprocess/string_to_index_1/Size<categorical_feature_preprocess/string_to_index_1/range/delta*

Tidx0*
_output_shapes
:
�
5categorical_feature_preprocess/string_to_index_1/CastCast6categorical_feature_preprocess/string_to_index_1/range*

SrcT0*
_output_shapes
:*

DstT0	
�
;categorical_feature_preprocess/string_to_index_1/hash_table	HashTable*
shared_name *
use_node_name_sharing( *
	key_dtype0*
	container *
value_dtype0	*
_output_shapes
:
�
Acategorical_feature_preprocess/string_to_index_1/hash_table/ConstConst*
valueB	 R
���������*
dtype0	*
_output_shapes
: 
�
Fcategorical_feature_preprocess/string_to_index_1/hash_table/table_initInitializeTable;categorical_feature_preprocess/string_to_index_1/hash_table6categorical_feature_preprocess/string_to_index_1/Const5categorical_feature_preprocess/string_to_index_1/Cast*N
_classD
B@loc:@categorical_feature_preprocess/string_to_index_1/hash_table*

Tval0	*

Tkey0
�
2categorical_feature_preprocess/hash_table_Lookup_1LookupTableFind;categorical_feature_preprocess/string_to_index_1/hash_tableExpandDims_6Acategorical_feature_preprocess/string_to_index_1/hash_table/Const*

Tout0	*N
_classD
B@loc:@categorical_feature_preprocess/string_to_index_1/hash_table*
_output_shapes

:
*	
Tin0
�
6categorical_feature_preprocess/string_to_index_2/ConstConst*:
value1B/BvanBcarBtrainBdroneBbikeBtruckB *
_output_shapes
:*
dtype0
w
5categorical_feature_preprocess/string_to_index_2/SizeConst*
value	B :*
_output_shapes
: *
dtype0
~
<categorical_feature_preprocess/string_to_index_2/range/startConst*
value	B : *
_output_shapes
: *
dtype0
~
<categorical_feature_preprocess/string_to_index_2/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
�
6categorical_feature_preprocess/string_to_index_2/rangeRange<categorical_feature_preprocess/string_to_index_2/range/start5categorical_feature_preprocess/string_to_index_2/Size<categorical_feature_preprocess/string_to_index_2/range/delta*

Tidx0*
_output_shapes
:
�
5categorical_feature_preprocess/string_to_index_2/CastCast6categorical_feature_preprocess/string_to_index_2/range*
_output_shapes
:*

DstT0	*

SrcT0
�
;categorical_feature_preprocess/string_to_index_2/hash_table	HashTable*
shared_name *
use_node_name_sharing( *
	key_dtype0*
	container *
value_dtype0	*
_output_shapes
:
�
Acategorical_feature_preprocess/string_to_index_2/hash_table/ConstConst*
valueB	 R
���������*
_output_shapes
: *
dtype0	
�
Fcategorical_feature_preprocess/string_to_index_2/hash_table/table_initInitializeTable;categorical_feature_preprocess/string_to_index_2/hash_table6categorical_feature_preprocess/string_to_index_2/Const5categorical_feature_preprocess/string_to_index_2/Cast*N
_classD
B@loc:@categorical_feature_preprocess/string_to_index_2/hash_table*

Tkey0*

Tval0	
�
2categorical_feature_preprocess/hash_table_Lookup_2LookupTableFind;categorical_feature_preprocess/string_to_index_2/hash_tableExpandDims_7Acategorical_feature_preprocess/string_to_index_2/hash_table/Const*

Tout0	*N
_classD
B@loc:@categorical_feature_preprocess/string_to_index_2/hash_table*
_output_shapes

:
*	
Tin0
�
bdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
�
adnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/CastCastbdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Shape*
_output_shapes
:*

DstT0	*

SrcT0
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast_1/xConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast_1Castednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0	
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/NotEqualNotEqual2categorical_feature_preprocess/hash_table_Lookup_1cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast_1*
T0	*
_output_shapes

:

�
bdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/WhereWhereednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/NotEqual*'
_output_shapes
:���������
�
jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/ReshapeReshape2categorical_feature_preprocess/hash_table_Lookup_1jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Reshape/shape*
T0	*
Tshape0*
_output_shapes
:

�
pdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"       
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stackrdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stack_1rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice/stack_2*
ellipsis_mask *

begin_mask*#
_output_shapes
:���������*
end_mask*
T0	*
Index0*
shrink_axis_mask*
new_axis_mask 
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
�
tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
�
tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1StridedSlicebdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Whererdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stacktdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1/stack_2*
end_mask*
ellipsis_mask *

begin_mask*
shrink_axis_mask *'
_output_shapes
:���������*
new_axis_mask *
T0	*
Index0
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/unstackUnpackadnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
�
bdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/stackPackfdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
�
`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/MulMulldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice_1bdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/stack*'
_output_shapes
:���������*
T0	
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/SumSum`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Mulrdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0	*#
_output_shapes
:���������
�
`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/AddAddjdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/strided_slice`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Sum*#
_output_shapes
:���������*
T0	
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/GatherGatherddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Reshape`dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Add*#
_output_shapes
:���������*
validate_indices(*
Tparams0	*
Tindices0	
�
Ndnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/mod/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
�
Ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/modFloorModcdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/GatherNdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/mod/y*
T0	*#
_output_shapes
:���������
�
idnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_sliceStridedSliceadnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Castidnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stackkdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stack_1kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice/stack_2*
end_mask *
ellipsis_mask *

begin_mask*
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
T0	*
Index0
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
�
mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1StridedSliceadnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Castkdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stackmdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stack_1mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1/stack_2*
end_mask*
ellipsis_mask *

begin_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
T0	*
Index0
�
[dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
Zdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/ProdProdednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_slice_1[dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat/values_1PackZdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/Prod*
_output_shapes
:*
N*

axis *
T0	
�
adnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
\dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concatConcatV2cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/strided_sliceednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat/values_1adnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat/axis*
_output_shapes
:*
N*
T0	*

Tidx0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshapeSparseReshapebdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Whereadnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/DenseToSparseTensor/Cast\dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/concat*-
_output_shapes
:���������:
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape/IdentityIdentityLdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/mod*
T0	*#
_output_shapes
:���������
�
_dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
valueB"      
�
^dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
valueB
 *    
�
`dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
valueB
 *���>
�
idnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal_dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
]dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/mulMulidnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormal`dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
T0
�
Ydnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normalAdd]dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/mul^dnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal/mean*
T0*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
<dnn/input_from_feature_columns/str2_embedding/weights/part_0
VariableV2*
shared_name *
shape
:*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
dtype0*
	container 
�
Cdnn/input_from_feature_columns/str2_embedding/weights/part_0/AssignAssign<dnn/input_from_feature_columns/str2_embedding/weights/part_0Ydnn/input_from_feature_columns/str2_embedding/weights/part_0/Initializer/truncated_normal*
_output_shapes

:*
validate_shape(*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
T0*
use_locking(
�
Adnn/input_from_feature_columns/str2_embedding/weights/part_0/readIdentity<dnn/input_from_feature_columns/str2_embedding/weights/part_0*
T0*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
�
idnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SliceSliceednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape:1jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice/beginidnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice/size*
_output_shapes
:*
Index0*
T0	
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/ProdProdddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Sliceddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
�
mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather/indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GatherGatherednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape:1mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather/indices*
_output_shapes
: *
validate_indices(*
Tparams0	*
Tindices0
�
vdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/new_shapePackcdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Prodednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather*
_output_shapes
:*
N*

axis *
T0	
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshapeSparseReshapecdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshapeednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape:1vdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:���������:
�
udnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/IdentityIdentityldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape/Identity*
T0	*#
_output_shapes
:���������
�
mdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GreaterEqualGreaterEqualudnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/Identitymdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GreaterEqual/y*
T0	*#
_output_shapes
:���������
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/WhereWherekdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/GreaterEqual*'
_output_shapes
:���������
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/ReshapeReshapeddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Whereldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape/shape*#
_output_shapes
:���������*
Tshape0*
T0	
�
gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_1Gatherldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshapefdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape*'
_output_shapes
:���������*
validate_indices(*
Tparams0	*
Tindices0	
�
gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_2Gatherudnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape/Identityfdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:���������
�
gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/IdentityIdentityndnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
�
xdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlicegdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Identity�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
end_mask *
Index0*
T0	*
shrink_axis_mask*
new_axis_mask 
�
wdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/CastCast�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice*
_output_shapes
: *

DstT0*

SrcT0	
�
~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
�
~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
�
xdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/rangeRange~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range/startwdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Cast~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range/delta*#
_output_shapes
:���������*

Tidx0
�
ydnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Cast_1Castxdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/range*

SrcT0*#
_output_shapes
:���������*

DstT0	
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlicegdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*
end_mask*

begin_mask*
ellipsis_mask *
shrink_axis_mask*#
_output_shapes
:���������*
new_axis_mask *
T0	*
Index0
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ListDiffListDiffydnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Cast_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlicegdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Identity�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDims�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/strided_slice_2�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*
_output_shapes
:*
T0	*

Tdim0
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
_output_shapes
: *
value	B
 Z
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDense{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ListDiff}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ExpandDims�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_values�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*
Tindices0	*
validate_indices(*
T0
*#
_output_shapes
:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
zdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ReshapeReshape{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ListDiff�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Reshape/shape*
T0	*'
_output_shapes
:���������*
Tshape0
�
}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:���������
�
~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
ydnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concatConcatV2zdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Reshape}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/zeros_like~dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat/axis*'
_output_shapes
:���������*
N*
T0	*

Tidx0
�
xdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ShapeShape{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/ListDiff*
_output_shapes
:*
out_type0*
T0	
�
wdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/FillFillxdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Shapexdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_1ydnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*
T0	*'
_output_shapes
:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Gather_2wdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/Fill�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_2/axis*
N*

Tidx0*
T0	*#
_output_shapes
:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorder{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_1{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/concat_2gdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Identity*
T0	*6
_output_shapes$
":���������:���������
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/IdentityIdentitygdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Identity*
_output_shapes
:*
T0	
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSlice�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseReorder�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*

begin_mask*
ellipsis_mask *
shrink_axis_mask*#
_output_shapes
:���������*
new_axis_mask *
Index0*
T0	
�
{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/CastCast�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:���������*

DstT0*

SrcT0	
�
}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/UniqueUnique�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseReorder:1*
out_idx0*
T0	*2
_output_shapes 
:���������:���������
�
�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherAdnn/input_from_feature_columns/str2_embedding/weights/part_0/read}dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/Unique*'
_output_shapes
:���������*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0*
Tparams0*
validate_indices(*
Tindices0	
�
vdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparseSparseSegmentMean�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/embedding_lookupdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/Unique:1{dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse/Cast*'
_output_shapes
:���������*
T0*

Tidx0
�
ndnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   
�
hdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_1Reshape�dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/SparseFillEmptyRows/SparseToDensendnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_1/shape*'
_output_shapes
:���������*
Tshape0*
T0

�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/ShapeShapevdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse*
_output_shapes
:*
out_type0*
T0
�
rdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
�
tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_sliceStridedSliceddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Shaperdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stacktdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stack_1tdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice/stack_2*
end_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
_output_shapes
: *
new_axis_mask *
Index0*
T0
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
�
ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/stackPackfdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/stack/0ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/TileTilehdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_1ddnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/stack*

Tmultiples0*
T0
*0
_output_shapes
:������������������
�
idnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/zeros_like	ZerosLikevdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
^dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweightsSelectcdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Tileidnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/zeros_likevdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
cdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/CastCastednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/InnerFlatten/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1Slicecdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Castldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1/beginkdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Shape_1Shape^dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights*
_output_shapes
:*
out_type0*
T0
�
ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
�
kdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2Slicefdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Shape_1ldnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2/beginkdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
ednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/concatConcatV2fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_1fdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Slice_2jdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_2Reshape^dnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweightsednn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/concat*'
_output_shapes
:���������*
Tshape0*
T0
�
`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/ShapeConst*
dtype0*
_output_shapes
:*
valueB"
      
�
_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/CastCast`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Shape*

SrcT0*
_output_shapes
:*

DstT0	
�
cdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast_1/xConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast_1Castcdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast_1/x*
_output_shapes
: *

DstT0	*

SrcT0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/NotEqualNotEqual0categorical_feature_preprocess/hash_table_Lookupadnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast_1*
T0	*
_output_shapes

:

�
`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/WhereWherecdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/NotEqual*'
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
bdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/ReshapeReshape0categorical_feature_preprocess/hash_table_Lookuphdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Reshape/shape*
_output_shapes
:
*
Tshape0*
T0	
�
ndnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
�
pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
�
pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
hdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Wherendnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stackpdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stack_1pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0	*#
_output_shapes
:���������*
shrink_axis_mask
�
pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
�
rdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
�
rdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1StridedSlice`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stackrdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stack_1rdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1/stack_2*
end_mask*

begin_mask*
ellipsis_mask *
shrink_axis_mask *'
_output_shapes
:���������*
new_axis_mask *
Index0*
T0	
�
bdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/unstackUnpack_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
�
`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/stackPackddnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/unstack:1*
_output_shapes
:*
N*

axis *
T0	
�
^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/MulMuljdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice_1`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/stack*'
_output_shapes
:���������*
T0	
�
pdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
�
^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/SumSum^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Mulpdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0	*#
_output_shapes
:���������
�
^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/AddAddhdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/strided_slice^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Sum*
T0	*#
_output_shapes
:���������
�
adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/GatherGatherbdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Reshape^dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Add*
Tindices0	*
validate_indices(*
Tparams0	*#
_output_shapes
:���������
�
Ldnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/mod/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
�
Jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/modFloorModadnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/GatherLdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/mod/y*#
_output_shapes
:���������*
T0	
�
gdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
idnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
idnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Castgdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stackidnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stack_1idnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice/stack_2*
end_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0	
�
idnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
�
kdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
cdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1StridedSlice_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Castidnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stackkdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stack_1kdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0	
�
Ydnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Xdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/ProdProdcdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slice_1Ydnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat/values_1PackXdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/Prod*
_output_shapes
:*
N*

axis *
T0	
�
_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Zdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concatConcatV2adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/strided_slicecdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat/values_1_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat/axis*
_output_shapes
:*
N*
T0	*

Tidx0
�
adnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshapeSparseReshape`dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/Where_dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/DenseToSparseTensor/CastZdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/concat*-
_output_shapes
:���������:
�
jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshape/IdentityIdentityJdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/mod*
T0	*#
_output_shapes
:���������
�
bdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SparseToDense/default_valueConst*
dtype0	*
_output_shapes
: *
valueB	 R
���������
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SparseToDenseSparseToDenseadnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshapecdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshape:1jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/InnerFlatten/SparseReshape/Identitybdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SparseToDense/default_value*0
_output_shapes
:������������������*
validate_indices(*
T0	*
Tindices0	
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Vdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
�
Wdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Xdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
Ndnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hotOneHotTdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SparseToDenseTdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/depthWdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/on_valueXdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot/off_value*
T0*4
_output_shapes"
 :������������������*
TI0	*
axis���������
�
\dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
Jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SumSumNdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/one_hot\dnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*'
_output_shapes
:���������
�
`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
      
�
_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/CastCast`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Shape*
_output_shapes
:*

DstT0	*

SrcT0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast_1Castcdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast_1/x*
_output_shapes
: *

DstT0	*

SrcT0
�
cdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/NotEqualNotEqual2categorical_feature_preprocess/hash_table_Lookup_2adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast_1*
_output_shapes

:
*
T0	
�
`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/WhereWherecdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/NotEqual*'
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
bdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/ReshapeReshape2categorical_feature_preprocess/hash_table_Lookup_2hdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Reshape/shape*
_output_shapes
:
*
Tshape0*
T0	
�
ndnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"       
�
pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
�
pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
hdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Wherendnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stackpdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stack_1pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice/stack_2*

begin_mask*
ellipsis_mask *#
_output_shapes
:���������*
end_mask*
Index0*
T0	*
shrink_axis_mask*
new_axis_mask 
�
pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
�
rdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
�
rdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
�
jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1StridedSlice`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Wherepdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stackrdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stack_1rdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1/stack_2*
end_mask*

begin_mask*
ellipsis_mask *
shrink_axis_mask *'
_output_shapes
:���������*
new_axis_mask *
Index0*
T0	
�
bdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/unstackUnpack_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Cast*
_output_shapes
: : *

axis *	
num*
T0	
�
`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/stackPackddnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/unstack:1*
_output_shapes
:*
N*

axis *
T0	
�
^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/MulMuljdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice_1`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/stack*
T0	*'
_output_shapes
:���������
�
pdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/SumSum^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Mulpdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0	*#
_output_shapes
:���������
�
^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/AddAddhdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/strided_slice^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Sum*#
_output_shapes
:���������*
T0	
�
adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/GatherGatherbdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Reshape^dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Add*#
_output_shapes
:���������*
validate_indices(*
Tparams0	*
Tindices0	
�
Ldnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/mod/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
�
Jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/modFloorModadnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/GatherLdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/mod/y*#
_output_shapes
:���������*
T0	
�
gdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
�
idnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
�
idnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Castgdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stackidnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stack_1idnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0	*
_output_shapes
:*
shrink_axis_mask 
�
idnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
�
kdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
�
kdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
cdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1StridedSlice_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Castidnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stackkdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stack_1kdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1/stack_2*
_output_shapes
:*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask *
Index0*
T0	
�
Ydnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Xdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/ProdProdcdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slice_1Ydnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat/values_1PackXdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/Prod*
_output_shapes
:*
N*

axis *
T0	
�
_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Zdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concatConcatV2adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/strided_slicecdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat/values_1_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat/axis*
N*

Tidx0*
T0	*
_output_shapes
:
�
adnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshapeSparseReshape`dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/Where_dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/DenseToSparseTensor/CastZdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/concat*-
_output_shapes
:���������:
�
jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshape/IdentityIdentityJdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/mod*
T0	*#
_output_shapes
:���������
�
bdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SparseToDense/default_valueConst*
dtype0	*
_output_shapes
: *
valueB	 R
���������
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SparseToDenseSparseToDenseadnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshapecdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshape:1jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/InnerFlatten/SparseReshape/Identitybdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SparseToDense/default_value*
Tindices0	*
validate_indices(*
T0	*0
_output_shapes
:������������������
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Vdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
�
Tdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
�
Wdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/on_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Xdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
Ndnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hotOneHotTdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SparseToDenseTdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/depthWdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/on_valueXdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot/off_value*4
_output_shapes"
 :������������������*
TI0	*
axis���������*
T0
�
\dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
�
Jdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/SumSumNdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/one_hot\dnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*'
_output_shapes
:���������
�
Ednn/input_from_feature_columns/input_from_feature_columns/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
@dnn/input_from_feature_columns/input_from_feature_columns/concatConcatV2hdnn/input_from_feature_columns/input_from_feature_columns/str2_embedding/str2_embeddingweights/Reshape_2Jdnn/input_from_feature_columns/input_from_feature_columns/str1_one_hot/SumJdnn/input_from_feature_columns/input_from_feature_columns/str3_one_hot/Sum numerical_feature_preprocess/add"numerical_feature_preprocess/add_1ExpandDims_4Ednn/input_from_feature_columns/input_from_feature_columns/concat/axis*
_output_shapes

:
*
N*
T0*

Tidx0
�
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"   
   
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�?�
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�?�>
�
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
seed2 *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*

seed *
_output_shapes

:
*
T0
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0
�
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0
�
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
_output_shapes

:
*
dtype0*
shape
:
*
	container *3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
_output_shapes

:
*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
use_locking(
�
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
dtype0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueB
*    
�
dnn/hiddenlayer_0/biases/part_0
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
_output_shapes
:
*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
use_locking(
�
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0
u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
_output_shapes

:
*
T0
�
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
_output_shapes

:

*
transpose_a( *
T0
o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes
:

�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*
_output_shapes

:

*
data_formatNHWC*
T0
p
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
_output_shapes

:

*
T0
W
zero_fraction/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    

zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*
_output_shapes

:


g
zero_fraction/CastCastzero_fraction/Equal*

SrcT0
*
_output_shapes

:

*

DstT0
d
zero_fraction/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
�
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values
�
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
�
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"
   
   
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�7�
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�7?
�
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
dtype0*

seed *
T0*
seed2 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0
�
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0
�
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
shared_name *
shape
:

*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
dtype0*
	container 
�
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:

*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*
_output_shapes
:
*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueB
*    
�
dnn/hiddenlayer_1/biases/part_0
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0
�
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
_output_shapes

:

*
T0
�
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
_output_shapes

:

*
transpose_a( *
T0
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:

�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*
_output_shapes

:

*
data_formatNHWC*
T0
p
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*
_output_shapes

:


Y
zero_fraction_1/zeroConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
_output_shapes

:

*
T0
k
zero_fraction_1/CastCastzero_fraction_1/Equal*

SrcT0
*
_output_shapes

:

*

DstT0
f
zero_fraction_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values
�
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
�
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
_output_shapes
: *
T0
�
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"
      
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��!�
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��!?
�
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
dtype0*

seed *
T0*
seed2 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*
T0*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*
T0*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
_output_shapes

:
*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
use_locking(
�
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*
_output_shapes

:
*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0
�
1dnn/hiddenlayer_2/biases/part_0/Initializer/ConstConst*
dtype0*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB*    
�
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name *
_output_shapes
:*
shape:
�
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
�
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
_output_shapes

:
*
T0
�
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *
_output_shapes

:
*
transpose_a( *
T0
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*
data_formatNHWC*
T0*
_output_shapes

:

p
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
_output_shapes

:
*
T0
Y
zero_fraction_2/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reluzero_fraction_2/zero*
T0*
_output_shapes

:

k
zero_fraction_2/CastCastzero_fraction_2/Equal*

SrcT0
*
_output_shapes

:
*

DstT0
f
zero_fraction_2/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_2_fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_2_fraction_of_zero_values
�
)dnn/hiddenlayer_2_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2_fraction_of_zero_values/tagszero_fraction_2/Mean*
_output_shapes
: *
T0
}
 dnn/hiddenlayer_2_activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_2_activation
�
dnn/hiddenlayer_2_activationHistogramSummary dnn/hiddenlayer_2_activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
_output_shapes
: *
T0
�
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      
�
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *׳]�
�
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *׳]?
�
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0*
dtype0*

seed *
T0*
seed2 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*
_output_shapes
: *,
_class"
 loc:@dnn/logits/weights/part_0*
T0
�
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0*
T0
�
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0*
T0
�
dnn/logits/weights/part_0
VariableV2*
shared_name *
shape
:*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0*
dtype0*
	container 
�
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
_output_shapes

:*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
use_locking(
�
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0*
T0
�
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    
�
dnn/logits/biases/part_0
VariableV2*
	container *
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name *
_output_shapes
:*
shape:
�
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
_output_shapes
:*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
use_locking(
�
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0*
T0
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
_output_shapes

:*
T0
�
dnn/logits/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/logits/weights*
transpose_b( *
_output_shapes

:
*
transpose_a( *
T0
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
_output_shapes
:*
T0
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*
data_formatNHWC*
T0*
_output_shapes

:

Y
zero_fraction_3/zeroConst*
_output_shapes
: *
dtype0*
valueB
 *    
q
zero_fraction_3/EqualEqualdnn/logits/BiasAddzero_fraction_3/zero*
T0*
_output_shapes

:

k
zero_fraction_3/CastCastzero_fraction_3/Equal*

SrcT0
*
_output_shapes

:
*

DstT0
f
zero_fraction_3/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
zero_fraction_3/MeanMeanzero_fraction_3/Castzero_fraction_3/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
'dnn/logits_fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *3
value*B( B"dnn/logits_fraction_of_zero_values
�
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
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
a
predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
T0*
_output_shapes

:

_
predictions/classes/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
�
predictions/classesArgMaxdnn/logits/BiasAddpredictions/classes/dimension*

Tidx0*
T0*
_output_shapes
:

�
0training_loss/softmax_cross_entropy_loss/SqueezeSqueeze+target_feature_preprocess/hash_table_Lookup*
T0	*
_output_shapes
:
*
squeeze_dims

x
.training_loss/softmax_cross_entropy_loss/ShapeConst*
_output_shapes
:*
dtype0*
valueB:

�
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*$
_output_shapes
:
:
*
Tlabels0	
]
training_loss/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
�
,metrics/remove_squeezable_dimensions/SqueezeSqueeze+target_feature_preprocess/hash_table_Lookup*
T0	*
_output_shapes
:
*
squeeze_dims

���������
~
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*
_output_shapes
:

Z
metrics/ToFloatCastmetrics/Equal*

SrcT0
*
_output_shapes
:
*

DstT0
[
metrics/accuracy/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    
z
metrics/accuracy/total
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
_output_shapes
: *
validate_shape(*)
_class
loc:@metrics/accuracy/total*
T0*
use_locking(
�
metrics/accuracy/total/readIdentitymetrics/accuracy/total*
T0*
_output_shapes
: *)
_class
loc:@metrics/accuracy/total
]
metrics/accuracy/zeros_1Const*
dtype0*
_output_shapes
: *
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
�
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
_output_shapes
: *
validate_shape(*)
_class
loc:@metrics/accuracy/count*
T0*
use_locking(
�
metrics/accuracy/count/readIdentitymetrics/accuracy/count*
T0*
_output_shapes
: *)
_class
loc:@metrics/accuracy/count
W
metrics/accuracy/SizeConst*
dtype0*
_output_shapes
: *
value	B :

i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*

SrcT0*
_output_shapes
: *

DstT0
`
metrics/accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*
use_locking( *
T0*
_output_shapes
: *)
_class
loc:@metrics/accuracy/total
�
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
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
T0*
_output_shapes
: 
]
metrics/accuracy/value/eConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
_output_shapes
: *
T0
a
metrics/accuracy/Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
_output_shapes
: *
T0
�
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
�
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
_output_shapes
: *
T0
N
metrics/RankConst*
dtype0*
_output_shapes
: *
value	B :
U
metrics/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B :
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
_output_shapes
: *
T0
�
metrics/Assert/ConstConst*
dtype0*
_output_shapes
: *N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]
�
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
�
metrics/Reshape/shapeConst^metrics/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������
�
metrics/ReshapeReshape+target_feature_preprocess/hash_table_Lookupmetrics/Reshape/shape*
T0	*
_output_shapes
:
*
Tshape0
]
metrics/one_hot/on_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
�
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
_output_shapes

:
*
TI0	*
axis���������*
T0
]
metrics/CastCastmetrics/one_hot*

SrcT0*
_output_shapes

:
*

DstT0

j
metrics/auc/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*
_output_shapes

:*
Tshape0*
T0
l
metrics/auc/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����
�
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
_output_shapes

:*
Tshape0*
T0

�
metrics/auc/ConstConst*
_output_shapes	
:�*
dtype0*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?
d
metrics/auc/ExpandDims/dimConst*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*
_output_shapes
:	�*
T0*

Tdim0
b
metrics/auc/stackConst*
dtype0*
_output_shapes
:*
valueB"      

metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*
_output_shapes
:	�
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
T0*
_output_shapes
: 
]
metrics/auc/transpose/sub/yConst*
_output_shapes
: *
dtype0*
value	B :
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
_output_shapes
: *
T0
c
!metrics/auc/transpose/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
T0*
_output_shapes

:
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
_output_shapes
:*
valueB"�      
�
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*
_output_shapes
:	�*
T0*

Tmultiples0
n
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*
_output_shapes
:	�
Z
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*
_output_shapes
:	�
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
_output_shapes
:*
valueB"�      
�
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
*
_output_shapes
:	�
[
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2*
_output_shapes
:	�
`
metrics/auc/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    
�
metrics/auc/true_positives
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
shape:�
�
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
_output_shapes	
:�*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
T0*
use_locking(
�
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*
T0*
_output_shapes	
:�*-
_class#
!loc:@metrics/auc/true_positives
n
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater*
_output_shapes
:	�
n
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*
_output_shapes
:	�*

DstT0*

SrcT0

c
!metrics/auc/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
�
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*
_output_shapes	
:�*-
_class#
!loc:@metrics/auc/true_positives*
T0*
use_locking( 
b
metrics/auc/zeros_1Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
metrics/auc/false_negatives
VariableV2*
shared_name *
dtype0*
shape:�*
_output_shapes	
:�*
	container 
�
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_negatives
�
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_negatives*
T0
s
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot*
_output_shapes
:	�
p
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

SrcT0
*
_output_shapes
:	�*

DstT0
e
#metrics/auc/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
�
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*
use_locking( *
T0*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_negatives
b
metrics/auc/zeros_2Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
metrics/auc/true_negatives
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
shape:�
�
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
_output_shapes	
:�*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
use_locking(
�
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*
_output_shapes	
:�*-
_class#
!loc:@metrics/auc/true_negatives*
T0
y
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot*
_output_shapes
:	�
p
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

SrcT0
*
_output_shapes
:	�*

DstT0
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*
use_locking( *
T0*
_output_shapes	
:�*-
_class#
!loc:@metrics/auc/true_negatives
b
metrics/auc/zeros_3Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
metrics/auc/false_positives
VariableV2*
shared_name *
dtype0*
shape:�*
_output_shapes	
:�*
	container 
�
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_positives
�
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_positives*
T0
v
metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater*
_output_shapes
:	�
p
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*
_output_shapes
:	�*

DstT0*

SrcT0

e
#metrics/auc/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
�
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*
_output_shapes	
:�*
T0*
	keep_dims( *

Tidx0
�
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*
use_locking( *
T0*
_output_shapes	
:�*.
_class$
" loc:@metrics/auc/false_positives
V
metrics/auc/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
_output_shapes	
:�*
T0
�
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:�
X
metrics/auc/add_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
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
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
_output_shapes	
:�*
T0
X
metrics/auc/add_4/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
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
i
metrics/auc/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
l
!metrics/auc/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:�
k
!metrics/auc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/strided_sliceStridedSlicemetrics/auc/div_1metrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
_output_shapes	
:�*
shrink_axis_mask 
k
!metrics/auc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
m
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes	
:�*
end_mask*
Index0*
T0*
shrink_axis_mask *
new_axis_mask 
t
metrics/auc/subSubmetrics/auc/strided_slicemetrics/auc/strided_slice_1*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
n
#metrics/auc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
m
#metrics/auc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
_output_shapes	
:�*
shrink_axis_mask 
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
_output_shapes	
:�*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask *
Index0*
T0
x
metrics/auc/add_5Addmetrics/auc/strided_slice_2metrics/auc/strided_slice_3*
_output_shapes	
:�*
T0
Z
metrics/auc/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
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
dtype0*
_output_shapes
:*
valueB: 
|
metrics/auc/valueSummetrics/auc/Mulmetrics/auc/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
metrics/auc/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
_output_shapes	
:�*
T0
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:�
X
metrics/auc/add_8/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:�
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
_output_shapes	
:�*
T0
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:�
Y
metrics/auc/add_10/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:�
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
_output_shapes	
:�*
T0
k
!metrics/auc/strided_slice_4/stackConst*
dtype0*
_output_shapes
:*
valueB: 
n
#metrics/auc/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
_output_shapes	
:�*
end_mask *
new_axis_mask *

begin_mask*
ellipsis_mask *
shrink_axis_mask *
Index0*
T0
k
!metrics/auc/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
m
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
m
#metrics/auc/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes	
:�*
end_mask*
Index0*
T0*
shrink_axis_mask *
new_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_4metrics/auc/strided_slice_5*
T0*
_output_shapes	
:�
k
!metrics/auc/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 
n
#metrics/auc/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
m
#metrics/auc/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes	
:�*
shrink_axis_mask 
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
_output_shapes
:*
valueB:
m
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
_output_shapes	
:�*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_6metrics/auc/strided_slice_7*
T0*
_output_shapes	
:�
\
metrics/auc/truediv_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
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
metrics/auc/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
�
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
*metrics/softmax_cross_entropy_loss/SqueezeSqueeze+target_feature_preprocess/hash_table_Lookup*
_output_shapes
:
*
T0	*
squeeze_dims

r
(metrics/softmax_cross_entropy_loss/ShapeConst*
_output_shapes
:*
dtype0*
valueB:

�
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*$
_output_shapes
:
:
*
Tlabels0	*
T0
a
metrics/eval_loss/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
W
metrics/mean/zerosConst*
dtype0*
_output_shapes
: *
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
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *%
_class
loc:@metrics/mean/total

metrics/mean/total/readIdentitymetrics/mean/total*
_output_shapes
: *%
_class
loc:@metrics/mean/total*
T0
Y
metrics/mean/zeros_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
v
metrics/mean/count
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
_output_shapes
: *
validate_shape(*%
_class
loc:@metrics/mean/count*
T0*
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
metrics/mean/ToFloat_1Castmetrics/mean/Size*

SrcT0*
_output_shapes
: *

DstT0
U
metrics/mean/ConstConst*
_output_shapes
: *
dtype0*
valueB 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*
_output_shapes
: *%
_class
loc:@metrics/mean/total*
T0*
use_locking( 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*
_output_shapes
: *%
_class
loc:@metrics/mean/count*
T0*
use_locking( 
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
metrics/mean/Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
metrics/mean/update_op/eConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
_output_shapes
: *
T0
`

group_depsNoOp^metrics/mean/update_op^metrics/auc/update_op^metrics/accuracy/update_op
\
eval_step/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
m
	eval_step
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
�
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
 *  �?
�
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@eval_step
�
initNoOp^global_step/AssignD^dnn/input_from_feature_columns/str2_embedding/weights/part_0/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign(^dnn/hiddenlayer_2/weights/part_0/Assign'^dnn/hiddenlayer_2/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_output_shapes
: *
dtype0	*
_class
loc:@global_step
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized<dnn/input_from_feature_columns/str2_embedding/weights/part_0*
dtype0*
_output_shapes
: *O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddnn/logits/weights/part_0*
_output_shapes
: *
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/logits/biases/part_0*
dtype0*
_output_shapes
: *+
_class!
loc:@dnn/logits/biases/part_0
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitialized"input_producer/limit_epochs/epochs*
dtype0	*
_output_shapes
: *5
_class+
)'loc:@input_producer/limit_epochs/epochs
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedmetrics/accuracy/total*
dtype0*
_output_shapes
: *)
_class
loc:@metrics/accuracy/total
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedmetrics/accuracy/count*
_output_shapes
: *
dtype0*)
_class
loc:@metrics/accuracy/count
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedmetrics/auc/true_positives*
dtype0*
_output_shapes
: *-
_class#
!loc:@metrics/auc/true_positives
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedmetrics/auc/false_negatives*
dtype0*
_output_shapes
: *.
_class$
" loc:@metrics/auc/false_negatives
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedmetrics/auc/true_negatives*
dtype0*
_output_shapes
: *-
_class#
!loc:@metrics/auc/true_negatives
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedmetrics/auc/false_positives*
_output_shapes
: *
dtype0*.
_class$
" loc:@metrics/auc/false_positives
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedmetrics/mean/total*
_output_shapes
: *
dtype0*%
_class
loc:@metrics/mean/total
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedmetrics/mean/count*
dtype0*
_output_shapes
: *%
_class
loc:@metrics/mean/count
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized	eval_step*
dtype0*
_output_shapes
: *
_class
loc:@eval_step
�	
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_19*
N*
T0
*
_output_shapes
:*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst*
dtype0*
_output_shapes
:*�
value�B�Bglobal_stepB<dnn/input_from_feature_columns/str2_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B"input_producer/limit_epochs/epochsBmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/countB	eval_step
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
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
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask 
�
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
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
_output_shapes
:*
Tshape0*
T0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
_output_shapes
:*
Tshape0
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
T0	*#
_output_shapes
:���������*
squeeze_dims

�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:���������
g
$report_uninitialized_resources/ConstConst*
_output_shapes
: *
dtype0*
valueB 
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*
T0*#
_output_shapes
:���������
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_output_shapes
: *
dtype0	*
_class
loc:@global_step
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized<dnn/input_from_feature_columns/str2_embedding/weights/part_0*
dtype0*
_output_shapes
: *O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: *
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddnn/logits/weights/part_0*
_output_shapes
: *
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0
�
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializeddnn/logits/biases/part_0*
dtype0*
_output_shapes
: *+
_class!
loc:@dnn/logits/biases/part_0
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_9*
_output_shapes
:
*
N
*

axis *
T0

}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:

�
&report_uninitialized_variables_1/ConstConst*
dtype0*
_output_shapes
:
*�
value�B�
Bglobal_stepB<dnn/input_from_feature_columns/str2_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
_output_shapes
:*
dtype0*
valueB:

�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
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
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:

�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
: *
end_mask*
Index0*
T0*
shrink_axis_mask *
new_axis_mask 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
T0*
_output_shapes
:
*
Tshape0
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
_output_shapes
:
*
Tshape0*
T0

�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
T0	*#
_output_shapes
:���������*
squeeze_dims

�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
Tindices0	*
validate_indices(*
Tparams0*#
_output_shapes
:���������
�
init_2NoOp*^input_producer/limit_epochs/epochs/Assign^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign^eval_step/Assign
�
init_all_tablesNoOp@^target_feature_preprocess/string_to_index/hash_table/table_initE^categorical_feature_preprocess/string_to_index/hash_table/table_initG^categorical_feature_preprocess/string_to_index_1/hash_table/table_initG^categorical_feature_preprocess/string_to_index_2/hash_table/table_init
/
group_deps_2NoOp^init_2^init_all_tables
�
Merge/MergeSummaryMergeSummary"input_producer/fraction_of_32_fullbatch/fraction_of_150_full)dnn/hiddenlayer_0_fraction_of_zero_valuesdnn/hiddenlayer_0_activation)dnn/hiddenlayer_1_fraction_of_zero_valuesdnn/hiddenlayer_1_activation)dnn/hiddenlayer_2_fraction_of_zero_valuesdnn/hiddenlayer_2_activation"dnn/logits_fraction_of_zero_valuesdnn/logits_activationtraining_loss/ScalarSummary*
_output_shapes
: *
N
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_691d66c29d724414a6f917008f91879d/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
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
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:
*�
value�B�
Bdnn/hiddenlayer_0/biasesBdnn/hiddenlayer_0/weightsBdnn/hiddenlayer_1/biasesBdnn/hiddenlayer_1/weightsBdnn/hiddenlayer_2/biasesBdnn/hiddenlayer_2/weightsB5dnn/input_from_feature_columns/str2_embedding/weightsBdnn/logits/biasesBdnn/logits/weightsBglobal_step
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:
*
dtype0*�
valuewBu
B10 0,10B21 10 0,21:0,10B10 0,10B10 10 0,10:0,10B5 0,5B10 5 0,10:0,5B7 3 0,7:0,3B3 0,3B5 3 0,5:0,3B 
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read%dnn/hiddenlayer_0/weights/part_0/read$dnn/hiddenlayer_1/biases/part_0/read%dnn/hiddenlayer_1/weights/part_0/read$dnn/hiddenlayer_2/biases/part_0/read%dnn/hiddenlayer_2/weights/part_0/readAdnn/input_from_feature_columns/str2_embedding/weights/part_0/readdnn/logits/biases/part_0/readdnn/logits/weights/part_0/readglobal_step*
dtypes
2
	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *'
_class
loc:@save/ShardedFilename*
T0
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
save/Const^save/control_dependency^save/MergeV2Checkpoints*
_output_shapes
: *
T0
|
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bdnn/hiddenlayer_0/biases
o
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueBB10 0,10
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
save/RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:*.
value%B#Bdnn/hiddenlayer_0/weights
y
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
dtype0*$
valueBB21 10 0,21:0,10
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assign dnn/hiddenlayer_0/weights/part_0save/RestoreV2_1*
_output_shapes

:
*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
use_locking(
~
save/RestoreV2_2/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bdnn/hiddenlayer_1/biases
q
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueBB10 0,10
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assigndnn/hiddenlayer_1/biases/part_0save/RestoreV2_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0

save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*.
value%B#Bdnn/hiddenlayer_1/weights
y
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*$
valueBB10 10 0,10:0,10
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assign dnn/hiddenlayer_1/weights/part_0save/RestoreV2_3*
_output_shapes

:

*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
use_locking(
~
save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*
dtype0*-
value$B"Bdnn/hiddenlayer_2/biases
o
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueBB5 0,5
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assigndnn/hiddenlayer_2/biases/part_0save/RestoreV2_4*
_output_shapes
:*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
use_locking(

save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*
dtype0*.
value%B#Bdnn/hiddenlayer_2/weights
w
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
dtype0*"
valueBB10 5 0,10:0,5
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assign dnn/hiddenlayer_2/weights/part_0save/RestoreV2_5*
_output_shapes

:
*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
use_locking(
�
save/RestoreV2_6/tensor_namesConst*
dtype0*
_output_shapes
:*J
valueAB?B5dnn/input_from_feature_columns/str2_embedding/weights
u
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
dtype0* 
valueBB7 3 0,7:0,3
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assign<dnn/input_from_feature_columns/str2_embedding/weights/part_0save/RestoreV2_6*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*O
_classE
CAloc:@dnn/input_from_feature_columns/str2_embedding/weights/part_0
w
save/RestoreV2_7/tensor_namesConst*
dtype0*
_output_shapes
:*&
valueBBdnn/logits/biases
o
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueBB3 0,3
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assigndnn/logits/biases/part_0save/RestoreV2_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*+
_class!
loc:@dnn/logits/biases/part_0
x
save/RestoreV2_8/tensor_namesConst*
dtype0*
_output_shapes
:*'
valueBBdnn/logits/weights
u
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
dtype0* 
valueBB5 3 0,5:0,3
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assigndnn/logits/weights/part_0save/RestoreV2_8*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*,
_class"
 loc:@dnn/logits/weights/part_0
q
save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBglobal_step
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2	
�
save/Assign_9Assignglobal_stepsave/RestoreV2_9*
use_locking(*
validate_shape(*
T0	*
_output_shapes
: *
_class
loc:@global_step
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard""
	eval_step

eval_step:0"
init_op

group_deps_1"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"�
model_variables�
�
>dnn/input_from_feature_columns/str2_embedding/weights/part_0:0
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0" 
global_step

global_step:0"�
table_initializer�
�
?target_feature_preprocess/string_to_index/hash_table/table_init
Dcategorical_feature_preprocess/string_to_index/hash_table/table_init
Fcategorical_feature_preprocess/string_to_index_1/hash_table/table_init
Fcategorical_feature_preprocess/string_to_index_2/hash_table/table_init"
ready_op


concat:0"�
dnn�
�
>dnn/input_from_feature_columns/str2_embedding/weights/part_0:0
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0"�
queue_runners��
�
input_producer)input_producer/input_producer_EnqueueMany#input_producer/input_producer_Close"%input_producer/input_producer_Close_1*
�
batch/fifo_queuebatch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/cond/Merge:0batch/fifo_queue_Close"batch/fifo_queue_Close_1*"!
local_init_op

group_deps_2"&

summary_op

Merge/MergeSummary:0"�
cond_context��
�
batch/cond/cond_textbatch/cond/pred_id:0batch/cond/switch_t:0 *�
ReaderReadUpToV2:0
ReaderReadUpToV2:1
batch/cond/control_dependency:0
*batch/cond/fifo_queue_EnqueueMany/Switch:1
,batch/cond/fifo_queue_EnqueueMany/Switch_1:1
,batch/cond/fifo_queue_EnqueueMany/Switch_2:1
batch/cond/pred_id:0
batch/cond/switch_t:0
batch/fifo_queue:0@
batch/fifo_queue:0*batch/cond/fifo_queue_EnqueueMany/Switch:1B
ReaderReadUpToV2:1,batch/cond/fifo_queue_EnqueueMany/Switch_2:1B
ReaderReadUpToV2:0,batch/cond/fifo_queue_EnqueueMany/Switch_1:1
�
batch/cond/cond_text_1batch/cond/pred_id:0batch/cond/switch_f:0*P
!batch/cond/control_dependency_1:0
batch/cond/pred_id:0
batch/cond/switch_f:0"�
	summaries�
�
$input_producer/fraction_of_32_full:0
batch/fraction_of_150_full:0
+dnn/hiddenlayer_0_fraction_of_zero_values:0
dnn/hiddenlayer_0_activation:0
+dnn/hiddenlayer_1_fraction_of_zero_values:0
dnn/hiddenlayer_1_activation:0
+dnn/hiddenlayer_2_fraction_of_zero_values:0
dnn/hiddenlayer_2_activation:0
$dnn/logits_fraction_of_zero_values:0
dnn/logits_activation:0
training_loss/ScalarSummary:0"�
trainable_variables��
�
>dnn/input_from_feature_columns/str2_embedding/weights/part_0:0Cdnn/input_from_feature_columns/str2_embedding/weights/part_0/AssignCdnn/input_from_feature_columns/str2_embedding/weights/part_0/read:0"C
5dnn/input_from_feature_columns/str2_embedding/weights  "
�
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weights
  "

�
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biases
 "

�
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weights

  "


�
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biases
 "

�
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weights
  "

�
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases "
�
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases ""J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"�
local_variables�
�
$input_producer/limit_epochs/epochs:0
metrics/accuracy/total:0
metrics/accuracy/count:0
metrics/auc/true_positives:0
metrics/auc/false_negatives:0
metrics/auc/true_negatives:0
metrics/auc/false_positives:0
metrics/mean/total:0
metrics/mean/count:0
eval_step:0"�
	variables��
7
global_step:0global_step/Assignglobal_step/read:0
�
>dnn/input_from_feature_columns/str2_embedding/weights/part_0:0Cdnn/input_from_feature_columns/str2_embedding/weights/part_0/AssignCdnn/input_from_feature_columns/str2_embedding/weights/part_0/read:0"C
5dnn/input_from_feature_columns/str2_embedding/weights  "
�
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weights
  "

�
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biases
 "

�
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weights

  "


�
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biases
 "

�
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weights
  "

�
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases "
�
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases "�5mGF       r5��	���7�A*9

loss1��?


auclx�>

global_step

accuracy/�h>�@i�