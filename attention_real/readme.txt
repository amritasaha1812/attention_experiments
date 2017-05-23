Step1
the intuition is first to compute the attention over the types given the input context
this is similar to the "Noise Mitigation for Neural Entity Typing and Relation Extraction" (https://arxiv.org/pdf/1612.07495.pdf) 

1. alpha^c_i,t = softmax(c_i M_c t)
2. a^c_t = sum_i(alpha_i,t, c_i)
3. a^c_t is of dimension R^h (h being hidden state of the RNN encoding the contexts)
4. a^c_t = softmax_t(a_t) this will give attention over types given the context

Step2
second to compute the attention over the types given the input mention sequence
this is also similar to the "Noise Mitigation for Neural Entity Typing and Relation Extraction" (https://arxiv.org/pdf/1612.07495.pdf)

1. alpha^m_i,t = softmax(m_i M_m t)
2. a^m_t = sum_i(alpha_i,t, m_i)
3. a^m_t is of dimension R^h (h being hidden state of the RNN encoding the contexts)
4. a^m_t = softmax_t(a_t) this will give attention over types given the context

Step3
third is to compute the attention over type t, while decoding the output sequence of types (i.e. decoding the path from the hierarchy)
A^c_tj = ((a^c_t h_j) (W_c t))       where A_tj is the attention over the t'th type when decoding the j'th step of the output RNN
				     a^c_t is the attention over the t'th type given the input context
				     h_j is the hidden representation from the j'th step of the output RNN
				     W_c is a parameter of dimension (#num_type_dim * #hidden_size) capturing the similarity between the latent type space and the hidden space of the RNN
				     t is the embedding of the type t

Similarly we have A^m_tj = ((a^m_t h_j) (W_m t))  	

A_tj = softmax(A^c_tj + A^m_tj)
A_tj = sum_t (A_tj t)

Now if we have hierarchical attention we will follow all of the above steps for the 0th level types (if we are following top-down attention) or L'th level types where L is the leaf level (if we are following bottom-up attention)

Only difference is that only types of that corresponding level would be considered when computing a^c_t or a^m_t

For the next level, the attention on the types is conditional to the attention on the type latent space previously discovered on its parent or child types (parent or child will depend on whether we took top-down or bottom-up attention)

Step3: (at the l+1'th level, where the previous level is denoted by l)
A^c_tj(l+1) = ((a^c_t h_j) (W_c (A_tjl t))) #A_tjl captures the attention over the latent type space as obtained from the previous level
A^m_tj(l+1) = ((a^m_t h_j) (W_m (A_tjl t)))

A_tj(l+1) = softmax(A^c_tj(l+1) + A^m_tj(l+1))
A_tj(l+1) = sum_t (A_tj(l+1)t)

