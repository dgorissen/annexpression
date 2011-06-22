% Copyright (c) 2009, Dirk Gorissen <dgorissen@gmail.com>
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the Surrogate Modeling Lab, Ghent University, INTEC-IBBT nor the names 
%       of its contributors may be used to endorse or promote products derived 
%       from this software without specific prior written permission.
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.

function s = getNeuralNetExpression(net, outputIndex)
% getNeuralNetExpression
%
% Description:
%     Return the symbolic expression for the output selected by outputIndex of the given
%	  Matlab Neural Network.
%
%     Note: only standard feed forward networks are supported
%        so no delays or combination functions other than netsum are supported.
%
%    More information about the Matlab Neural Network object makeup can be found here:
%	 http://www.mathworks.com/help/toolbox/nnet/ug/bss4hk6-1.html

if(~exist('outputIndex'))
	outputIndex = 1;
end

% Get the expression for the given output neuron
% The final expression is recursively built up
s = expressionForNeuron(net,net.numLayers,outputIndex,'');

	% calculate the expression for neuron number 'neuron' in layer 'layer'
	function s = expressionForNeuron(net,layer,neuron,s)
		% should the transfer function names be replaced by their equations
		flattenTfunc = 1;
		% precision for the weights and biases
		precision = '%0.6d';
		
		% base case of the recursion, the input layer
		if(layer == 1)
			% each input is denoted by xi
			str = '';
			% get the matrix of weights from the input layer to the inputs
			L = net.IW{layer,1};
			% NOTE: we only consider the netsum function (neuron activation is a weighted sum of inputs)
			for i=1:net.inputs{layer}.size;
				w = L(neuron,i);
				input = [ 'x' num2str(i)];
				str = [str input '*'  sprintf(precision,w) ' + '];
				%str = [str input '*'  num2str(w) ' + '];
			end
			% dont forget to add the bias
			biases = net.b{layer};
			bias = biases(neuron);
			str = [str sprintf(precision,bias)];
			%str = [str num2str(bias)];
			if(flattenTfunc)
				str = ['(' expressionForTfunc(net.layers{layer}.transferFcn,str) ')'];
			else
				str = [net.layers{layer}.transferFcn '(' str ')'];
			end
			s = str;
			%disp(sprintf('Input neuron %i: %s',neuron,s));
		else
			% calculate the expression for one of the hidden or output neurons
			fanIn = net.layers{layer-1}.size;
			L = net.LW{layer,layer-1};
			s = '';
			% again we only consider a weighted sum of inputs
			for i=1:fanIn
				w = L(neuron,i);
				% recurively call the function for each neuron in the fan-in
				input = expressionForNeuron(net,layer-1,i,s);
				s = [s input '*'  sprintf(precision,w) ' + '];
				%s = [s input '*'  num2str(w) ' + '];
			end
			% dont forget to add the bias
			biases = net.b{layer};
			bias = biases(neuron);
			s = [s sprintf(precision,bias)];
			%s = [s num2str(bias)];
			if(flattenTfunc)
				s = ['(' expressionForTfunc(net.layers{layer}.transferFcn,s) ')'];
			else
				s = [net.layers{layer}.transferFcn '(' s ')'];
			end
			%disp(sprintf('Layer %i neuron %i: %s',layer,neuron,s));
		end
	end

	% given a name of a matlab neural network transfer function, return the mathematical expression
	% applied to the given input argument s
	function expr = expressionForTfunc(tfunc,s)
		if(strcmp(tfunc,'tansig'))
			% definition: a = tansig(n) = 2/(1+exp(-2*n))-1
			expr = ['2/(1+exp(-2*(' s ')))-1'];
		elseif(strcmp(tfunc,'logsig'))
			% definition: logsig(n) = 1 / (1 + exp(-n))
			expr = ['1/(1+exp(-(' s ')))'];
		elseif(strcmp(tfunc,'radbas'))
			% definition: exp(-n^2)
			expr = ['exp(-(' s ')^2)'];
		elseif(strcmp(tfunc,'purelin'))
			expr = ['(' s ')'];
		else
			error('Unsupported transfer function');
		end
	end

end
