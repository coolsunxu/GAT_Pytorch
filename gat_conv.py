import torch
import torch.nn as nn


class BatchMultiHeadGraphAttention(nn.Module): # 多头图注意力模型
	def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
		super(BatchMultiHeadGraphAttention, self).__init__()
		self.n_head = n_head # 头大小
		self.f_in = f_in # 输入大小
		self.f_out = f_out # 输出大小
		self.attn_dropout = attn_dropout # dropout
		self.add_self_loop = True # 为防止没有邻居结点出现的情况
		self.w = nn.Parameter(torch.Tensor(self.n_head, self.f_in, self.f_out)) # 自定义参数 权重
		
		#self.fc = nn.Linear(self.f_out*2,1) # 求分数,可能要修改 多个头参数
		self.fc = nn.Parameter(torch.Tensor(self.n_head, self.f_out*2,1)) # 自定义参数 att
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2) # 激活函数
		self.softmax = nn.Softmax(dim=-1) # 归一层
		self.dropout = nn.Dropout(self.attn_dropout) # Dropout 层
		if bias:
			self.bias = nn.Parameter(torch.Tensor(f_out)) # 自定义参数 偏置
			nn.init.constant_(self.bias, 0) # 初始化参数
		else:
			self.register_parameter("bias", None)

		# 初始化自定义参数
		nn.init.xavier_uniform_(self.w, gain=1.414)
		nn.init.xavier_uniform_(self.fc, gain=1.414)
	
	def remove_self_loops(self,edge_index): # 移除自环
		row, col = edge_index
		mask = row != col # 返回的是序号 不相等
		edge_index = edge_index[:, mask]
		return edge_index
	
	def add_self_loops(self, edge_index, num_nodes): # 添加自环
		loop_index = torch.arange(0, num_nodes, dtype=torch.long,
								  device=edge_index.device)
		loop_index = loop_index.unsqueeze(0).repeat(2, 1)
		edge_index = torch.cat([edge_index, loop_index], dim=1)
		return edge_index

	def forward(self, h, edge_index): 
		bs = h.shape[0] # [bs]
		if self.add_self_loop: # 是否添加自环 
			self.remove_self_loops(edge_index)
			self.add_self_loops(edge_index, bs)
		self.adj = torch.zeros(self.n_head,bs,bs).to(h) # [head,bs,bs] 邻接矩阵
		
		h_prime = torch.matmul(h, self.w) # [head,bs,fout]
		
		for i in range(h_prime.shape[1]): # for each node
			neighbors = edge_index[1][(edge_index[0,:]==i).nonzero().squeeze(0)] # neighbors
			n_neighbors = neighbors.shape[0] # number of this node's neighbors
			curr_node = h_prime[:,i,:].unsqueeze(1).repeat(1, n_neighbors, 1) # [head,cbs,fout]
			neighbors_node = h_prime[:,neighbors,:] # [head,cbs,fout]
			total_node = torch.cat((curr_node,neighbors_node),2) # [head,cbs,fout*2]
			
			att_node = self.leaky_relu(torch.matmul(total_node,self.fc))
			att_node = self.softmax(att_node.reshape(self.n_head,n_neighbors)) # [head,cbs]
			att_node = self.dropout(att_node)
			for k,v in enumerate(neighbors):
				self.adj[:,i,v] = att_node[:,k]
				
		output = torch.matmul(self.adj,h_prime)  # [head,bs,f_out]
		output = torch.mean(output,0) # [bs,fout]
		
		if self.bias is not None:
			return output + self.bias
		else:
			return output

# 生成领接矩阵
def Get_Adj(bs):
	import itertools
	a = [[],[]]
	list_indices = torch.arange(0, bs) # 行人列表
	for i, j in itertools.permutations(list_indices, 2): # 行人-行人 两两全排列
		a[0].append(i)
		a[1].append(j)
	return torch.LongTensor(a)
	
import time
heads = 12
bs = 512
fin = 256
fout = 128
a = Get_Adj(bs)
h = torch.randn(bs,fin)
model = BatchMultiHeadGraphAttention(n_head=heads, f_in=fin, f_out=fout, attn_dropout=0.5)
start_time = time.time()  #开始时间
out = model(h,a)
end_time = time.time()   #结束时间
print("time:%d"  % (end_time-start_time))
print(out.shape)
