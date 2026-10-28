# Migracao-Backend
Migração do backend do TutorIA, de Ollama a vLLM



O mecanismo de atenção em LLMs de arquitetura Transformer possuiria baixa latência sem o uso de método de otimização. Pela natureza Transformer, há a inferência do próximo token com base nos tokens anteriores e para gerar tokens relevantes aos inputs anteriores, ocorre o que se tem por Self-Attention, que avalia o peso de cada token na geração do seguinte, utilizando os três vetores provenientes de um token: $$Query (Q)$$, $$Key (K)$$ e $$Value (V)$$. 
