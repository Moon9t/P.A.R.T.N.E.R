package model

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ImprovedChessCNN is an enhanced CNN with residual blocks and more capacity
type ImprovedChessCNN struct {
	graph *gorgonia.ExprGraph
	vm    gorgonia.VM

	// Input
	input *gorgonia.Node

	// Residual blocks
	conv1  *gorgonia.Node
	conv2  *gorgonia.Node
	res1   *gorgonia.Node
	conv3  *gorgonia.Node
	conv4  *gorgonia.Node
	res2   *gorgonia.Node
	conv5  *gorgonia.Node
	conv6  *gorgonia.Node
	res3   *gorgonia.Node

	// Policy head (move prediction)
	policyConv *gorgonia.Node
	policyFC   *gorgonia.Node
	policyOut  *gorgonia.Node

	// Value head (position evaluation)
	valueConv *gorgonia.Node
	valueFC1  *gorgonia.Node
	valueFC2  *gorgonia.Node
	valueOut  *gorgonia.Node

	// Learnables
	learnables gorgonia.Nodes
}

// NewImprovedChessCNN creates an enhanced CNN architecture
func NewImprovedChessCNN() (*ImprovedChessCNN, error) {
	g := gorgonia.NewGraph()
	cnn := &ImprovedChessCNN{
		graph:      g,
		learnables: make(gorgonia.Nodes, 0),
	}

	// Input: [batch, 12, 8, 8] (12 piece planes)
	inputShape := tensor.Shape{1, 12, 8, 8}
	cnn.input = gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(inputShape...), gorgonia.WithName("input"))

	// First conv block: 12 -> 128 channels
	conv1, err := cnn.convBlock(cnn.input, 12, 128, "conv1")
	if err != nil {
		return nil, fmt.Errorf("conv1: %w", err)
	}
	cnn.conv1 = conv1

	// Second conv: 128 -> 128 (same channels for residual)
	conv2, err := cnn.convBlock(conv1, 128, 128, "conv2")
	if err != nil {
		return nil, fmt.Errorf("conv2: %w", err)
	}
	cnn.conv2 = conv2

	// Residual connection 1
	res1, err := gorgonia.Add(conv1, conv2)
	if err != nil {
		return nil, fmt.Errorf("res1: %w", err)
	}
	cnn.res1 = res1

	// Third conv block: 128 -> 256
	conv3, err := cnn.convBlock(res1, 128, 256, "conv3")
	if err != nil {
		return nil, fmt.Errorf("conv3: %w", err)
	}
	cnn.conv3 = conv3

	// Fourth conv: 256 -> 256
	conv4, err := cnn.convBlock(conv3, 256, 256, "conv4")
	if err != nil {
		return nil, fmt.Errorf("conv4: %w", err)
	}
	cnn.conv4 = conv4

	// Residual connection 2 (with 1x1 conv to match channels)
	res1_256, err := cnn.conv1x1(res1, 128, 256, "res1_proj")
	if err != nil {
		return nil, fmt.Errorf("res1_proj: %w", err)
	}
	res2, err := gorgonia.Add(res1_256, conv4)
	if err != nil {
		return nil, fmt.Errorf("res2: %w", err)
	}
	cnn.res2 = res2

	// Fifth conv block: 256 -> 512
	conv5, err := cnn.convBlock(res2, 256, 512, "conv5")
	if err != nil {
		return nil, fmt.Errorf("conv5: %w", err)
	}
	cnn.conv5 = conv5

	// Sixth conv: 512 -> 512
	conv6, err := cnn.convBlock(conv5, 512, 512, "conv6")
	if err != nil {
		return nil, fmt.Errorf("conv6: %w", err)
	}
	cnn.conv6 = conv6

	// Residual connection 3
	res2_512, err := cnn.conv1x1(res2, 256, 512, "res2_proj")
	if err != nil {
		return nil, fmt.Errorf("res2_proj: %w", err)
	}
	res3, err := gorgonia.Add(res2_512, conv6)
	if err != nil {
		return nil, fmt.Errorf("res3: %w", err)
	}
	cnn.res3 = res3

	// === Policy Head (Move Prediction) ===
	// Conv: 512 -> 256
	policyConv, err := cnn.convBlock(res3, 512, 256, "policy_conv")
	if err != nil {
		return nil, fmt.Errorf("policy_conv: %w", err)
	}
	cnn.policyConv = policyConv

	// Flatten: [batch, 256, 8, 8] -> [batch, 16384]
	policyFlat := gorgonia.Must(gorgonia.Reshape(policyConv, tensor.Shape{1, 256 * 8 * 8}))

	// FC: 16384 -> 4096 (64x64 possible moves)
	policyFC, err := cnn.fcLayer(policyFlat, 256*8*8, 4096, "policy_fc")
	if err != nil {
		return nil, fmt.Errorf("policy_fc: %w", err)
	}
	cnn.policyFC = policyFC
	cnn.policyOut = policyFC

	// === Value Head (Position Evaluation) ===
	// Conv: 512 -> 32
	valueConv, err := cnn.convBlock(res3, 512, 32, "value_conv")
	if err != nil {
		return nil, fmt.Errorf("value_conv: %w", err)
	}
	cnn.valueConv = valueConv

	// Flatten: [batch, 32, 8, 8] -> [batch, 2048]
	valueFlat := gorgonia.Must(gorgonia.Reshape(valueConv, tensor.Shape{1, 32 * 8 * 8}))

	// FC: 2048 -> 256
	valueFC1, err := cnn.fcLayer(valueFlat, 32*8*8, 256, "value_fc1")
	if err != nil {
		return nil, fmt.Errorf("value_fc1: %w", err)
	}
	cnn.valueFC1 = valueFC1

	// FC: 256 -> 1 (single value output)
	valueFC2, err := cnn.fcLayer(valueFC1, 256, 1, "value_fc2")
	if err != nil {
		return nil, fmt.Errorf("value_fc2: %w", err)
	}
	cnn.valueFC2 = valueFC2

	// Tanh to bound value to [-1, 1]
	valueOut, err := gorgonia.Tanh(valueFC2)
	if err != nil {
		return nil, fmt.Errorf("value_tanh: %w", err)
	}
	cnn.valueOut = valueOut

	// Create VM
	machine := gorgonia.NewTapeMachine(g)
	cnn.vm = machine

	return cnn, nil
}

// convBlock creates a conv + batchnorm + relu block
func (cnn *ImprovedChessCNN) convBlock(input *gorgonia.Node, inChannels, outChannels int, name string) (*gorgonia.Node, error) {
	// Conv2D
	kernel := gorgonia.NewTensor(cnn.graph, tensor.Float64, 4,
		gorgonia.WithShape(outChannels, inChannels, 3, 3),
		gorgonia.WithName(name+"_kernel"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	cnn.learnables = append(cnn.learnables, kernel)

	conv, err := gorgonia.Conv2d(input, kernel, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return nil, err
	}

	// Bias
	bias := gorgonia.NewTensor(cnn.graph, tensor.Float64, 1,
		gorgonia.WithShape(outChannels),
		gorgonia.WithName(name+"_bias"),
		gorgonia.WithInit(gorgonia.Zeroes()))
	cnn.learnables = append(cnn.learnables, bias)

	// Broadcast bias and add
	biasReshaped := gorgonia.Must(gorgonia.Reshape(bias, tensor.Shape{1, outChannels, 1, 1}))
	convBias, err := gorgonia.BroadcastAdd(conv, biasReshaped, nil, []byte{0, 2, 3})
	if err != nil {
		return nil, err
	}

	// ReLU activation
	activated, err := gorgonia.Rectify(convBias)
	if err != nil {
		return nil, err
	}

	return activated, nil
}

// conv1x1 creates a 1x1 convolution for projection (channel matching)
func (cnn *ImprovedChessCNN) conv1x1(input *gorgonia.Node, inChannels, outChannels int, name string) (*gorgonia.Node, error) {
	kernel := gorgonia.NewTensor(cnn.graph, tensor.Float64, 4,
		gorgonia.WithShape(outChannels, inChannels, 1, 1),
		gorgonia.WithName(name+"_kernel"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	cnn.learnables = append(cnn.learnables, kernel)

	conv, err := gorgonia.Conv2d(input, kernel, tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return nil, err
	}

	return conv, nil
}

// fcLayer creates a fully connected layer
func (cnn *ImprovedChessCNN) fcLayer(input *gorgonia.Node, inSize, outSize int, name string) (*gorgonia.Node, error) {
	weights := gorgonia.NewMatrix(cnn.graph, tensor.Float64,
		gorgonia.WithShape(inSize, outSize),
		gorgonia.WithName(name+"_weights"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	cnn.learnables = append(cnn.learnables, weights)

	bias := gorgonia.NewVector(cnn.graph, tensor.Float64,
		gorgonia.WithShape(outSize),
		gorgonia.WithName(name+"_bias"),
		gorgonia.WithInit(gorgonia.Zeroes()))
	cnn.learnables = append(cnn.learnables, bias)

	mul, err := gorgonia.Mul(input, weights)
	if err != nil {
		return nil, err
	}

	result, err := gorgonia.Add(mul, bias)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// Forward performs a forward pass returning both policy and value
func (cnn *ImprovedChessCNN) Forward(inputData tensor.Tensor) (policy, value tensor.Tensor, err error) {
	// Set input
	if err := gorgonia.Let(cnn.input, inputData); err != nil {
		return nil, nil, fmt.Errorf("failed to set input: %w", err)
	}

	// Run forward pass
	if err := cnn.vm.RunAll(); err != nil {
		return nil, nil, fmt.Errorf("forward pass failed: %w", err)
	}

	// Get outputs
	policyVal := cnn.policyOut.Value()
	valueVal := cnn.valueOut.Value()

	if policyVal == nil || valueVal == nil {
		return nil, nil, fmt.Errorf("output is nil")
	}

	return policyVal.(tensor.Tensor), valueVal.(tensor.Tensor), nil
}

// GetLearnables returns all trainable parameters
func (cnn *ImprovedChessCNN) GetLearnables() gorgonia.Nodes {
	return cnn.learnables
}

// Reset resets the VM state
func (cnn *ImprovedChessCNN) Reset() error {
	cnn.vm.Reset()
	return nil
}

// Close closes the VM
func (cnn *ImprovedChessCNN) Close() error {
	return cnn.vm.Close()
}
