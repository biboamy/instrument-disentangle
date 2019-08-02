from block import *

class Net(nn.Module):
	def __init__(self, name):
		super(Net, self).__init__()
		if 'UnetAE' in name:
			self.encode = Encode(name)
		if 'DuoAE' in name:
			self.pitch_encode = Encode(name)
			self.inst_encode = Encode(name)
		if 'preI' in name:
			self.inst_decode = InstDecoder()
		if 'preP' in name:
			self.pitch_decode = PitchDecoder()
		if 'preRoll' in name:
			self.roll_decode = Decode(name)

	def forward(self, _input, Xavg, Xstd, name, trainAdv):
		def get_inst_x(x,avg,std):
			xs = x.size()
			avg = avg.view(1, avg.size()[0],1,1).repeat(xs[0], 1, xs[2], 1).type('torch.cuda.FloatTensor')
			std = std.view(1, std.size()[0],1,1).repeat(xs[0], 1, xs[2], 1).type('torch.cuda.FloatTensor')
			x = (x - avg)/std
			return x
		
		x = _input.unsqueeze(3) 
		x = get_inst_x(x,Xavg,Xstd)
		x = x.permute(0,3,1,2)

		l1,l2,l3,l4,l5 = torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1)
		if 'UnetAE' in name:
			vec,s,c = self.encode(x,name)
			if not trainAdv and 'preI' in name:
				l1 = self.inst_decode(vec) # predict inst
			if not trainAdv and 'prePP' in name:
				l2 = self.pitch_decode(vec.detach())
			if trainAdv and 'prePN' in name:
				l4 = self.pitch_decode(vec) # predict pitch
			if not trainAdv and 'preRoll' in name:
				l5 = self.roll_decode(vec,name,s,c) # predict roll
			vec = vec
		
		if 'DuoAE' in name:
			vec_p, s = self.pitch_encode(x,name)
			vec_i, s = self.inst_encode(x,name)
			if not trainAdv and 'preIP' in name:
				l1 = self.inst_decode(vec_i) # predict inst
			if not trainAdv and 'prePP' in name:
				l2 = self.pitch_decode(vec_p) # predict pitch
			if trainAdv and 'preIN' in name:
				l3 = self.inst_decode(vec_p) # predict inst neg
			if trainAdv and 'prePN' in name:
				l4 = self.pitch_decode(vec_i) # predict pitch neg
			if not trainAdv and 'preRoll' in name:
				l5 = self.roll_decode(torch.cat((vec_i,vec_p),1),name,s) # predict roll
			vec = vec_i
	
		return l1,l2,l3,l4,l5,vec
