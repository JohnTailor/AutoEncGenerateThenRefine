Key code passages:

Changes to model.py in https://github.com/karpathy/nanoGPT:
---------------------------------
#We effectively take only the l-th, e.g., 4-th output; this could be made more efficient with custom attention masks
def forward(self, idx, targets=None):
	device = idx.device
	b, t = idx.size()
	assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
	pos = torch.arange(0, t , dtype=torch.long, device=device)  # shape (t)
	pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
	tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

	# forward the GPT model itself
	x = self.transformer.drop(tok_emb + pos_emb)
	for block in self.transformer.h: x = block(x)
	x = self.transformer.ln_f(x)
	x=x[:,self.cauTok-1::self.cauTok] #self.cauTok == l = 4 in paper
	if targets is not None: # if we are given some desired targets also calculate the loss
		logits = self.lm_head(x)
		loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
	else: # inference-time mini-optimization: only forward the lm_head on the very last position
		logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
		loss = None
	return logits, loss


Training:   
  # getting a batch through considering subsequences and permuting given xgap,ygap and regular non-permuted batch x,y
    bs = cfg["block_size"]
    catok = cfg["cauTok"] #caTok = l = 4 in paper
    def get_batch(split,off=1,gen=None,valfac=3): # We recreate np.memmap every batch to avoid a memory leak, as per https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data,bas=(datatr,cfg["batch_size"]) if split=="train" else (datava,valfac*cfg["batch_size"]) #((datavaall,valfac*cfg["batch_size"]) if split=="valall" else ((datavawo,valfac*cfg["batch_size"]) if split=="valwo" else (datafew,valfac*cfg["batch_size"]))) #(None,None)
        ix = torch.randint(len(data) - bs - 1, (bas,),generator=gen)
        all= torch.stack([torch.from_numpy((data[i  : i + off+ bs]).astype(np.int64)) for i in ix])
        with torch.no_grad():
            allgap=all.clone()
            torep=allgap[:,catok-1::catok]
            torep1=allgap[:,catok::catok].clone()
            allgap[:,catok::catok]=torep
            allgap[:,catok - 1::catok]=torep1
            xgap = allgap[:, :bs]
            ygap = allgap[:, catok::catok]
            x=all[:,:bs]
            y=all[:,catok::catok]
        x, y = x.pin_memory().to(cudstr, non_blocking=True), y.pin_memory().to(cudstr, non_blocking=True)  # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        xgap, ygap = xgap.pin_memory().to(cudstr, non_blocking=True), ygap.pin_memory().to(cudstr, non_blocking=True)  # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        return x, y,xgap,ygap


Algorithm AGR: Note we use logits rather than softmax outputs
-------------------------

    #Core of Algorithm AGR in paper, in our vanilla configuration we use k=15, w=0.05
    getco = lambda log, cy: np.sum((torch.argmax(log, dim=-1).view(-1) == cy).cpu().numpy())
    totRes={}
    for split in ["train","val"]:
      iter_num= 0
      tloNe, tloGap = 0, 0
      tot, coGM, coGap, coNe, coNeTop = 0, 0, 0, 0, 0

      while iter_num < cfg["eval_iters"]:
        x, oy, xg, oyg = get_batch(split)
        y=oy[:, -1]
        yg=oyg[:,-1]
        iter_num += 1
        with torch.no_grad():
            ologgap, _ = modelGap(xg) #Model f_s in paper
            coGap += getco(ologgap, yg)
            ologgap, loGap = modelGap(xg,oyg)
            tloGap+=torch.mean(loGap).item()

            # ptokga=torch.argmax(ologgap[0],-1)
            # print("xGap:",decode(xg[0][:cauTok+1].cpu().numpy())) #[-cauTok:]
            # ptok=torch.argmax(ologgap[0],-1)
            # print("yGap tr/pr:", decode(oyg[0].cpu().numpy()[:1]),decode(ptok.cpu().numpy()[:1])) #x[0][cfg["cauTok"]::cfg["cauTok"]]

            logne, loNe = modelNe(x,oy) #Model f_n in paper
            tloNe+=torch.mean(loNe).item()
            logne=logne[:,-1,:]
            top_values, top_indices = torch.topk(logne, k=k, dim=-1)
            top_indices=top_indices.squeeze()
            
            #top_values=top_values.squeeze()            #print(top_indices[0].cpu().numpy())            #ptokNe = torch.argmax(logne, -1)
            #print("topNe:", decode(top_indices[0].cpu().numpy()))            #print("topVal",top_values[0,0], "me",torch.mean(top_values[0]).item())            #print("logTrue",logne[0,y[0]])            #print(top_indices.shape,xg.shape)
            
            if 1:#wgap>0: #in paper wgap = 99 - it is not used
                loggap=[]
                for k in range(k):
                    xgt=xg.clone()
                    xgt[:, -1] = top_indices[:,k]
                    cloggap, closs = modelGap(xgt)
                    cloggap=cloggap.squeeze()
                    ctokGap = torch.argmax(cloggap, -1)
                    corr = ctokGap.view(-1) == yg
                    #if mul==0: vals = (1 - wgap) * top_values[:, k] + wgap * corr.float()*w  else:
                    vals =  top_values[:, k] *(1+ corr.float() * w)                    
                    loggap.append(vals)
                loggap=torch.stack(loggap,dim=1)                
                imtoks = torch.argmax(loggap, -1)                
                mtoks = top_indices[np.arange(len(top_indices)), imtoks]
                # decw=[decode([w]) for w in top_indices[0].cpu().numpy()]
                # print("GapNe:", list(zip(np.round(allmtoks[0].cpu().numpy(), 1), decw)))
                # print("GapNe co?", (y[0] == mtoks[0]).item())
                # print("Ne co?", (y[0] == ptok[0]).item())
                # print("Gap co?", (yg[0] == ptokga[-1]).item())
                coGM+=np.sum((mtoks==y).cpu().numpy())



            tot+=len(y)
            coNe+=getco(logne,y)
            for k in range(len(top_indices)):
                if y[k] in top_indices[k]:coNeTop+=1