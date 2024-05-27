# Embedding Generative Tuning Language Model

Embedding Generative Tuning Language Model (EGTLM), leveraging the robust semantic processing capabilities of large language models. To enhance the model's multitasking abilities, a two-stage training approach involving unsupervised pre-training followed by supervised instruction tuning is proposed. The resultant EGTLM model excels in both embedding and text generation. An RAG knowledge system solution based on the EGTLM model is also designed. Remarkably, the EGTLM model achieved a score of 68 in the Chinese evaluation rankings of the Massive Text Embedding Benchmark, securing a spot in the top ten. 

## Main Highlights

To train the model, a mixed-task approach is used. The loss functions involved are as follows:

The generative loss function, \(\mathcal{L}_{Gen}\), is defined as:

$$
\mathcal{L}_{Gen} = -\frac{1}{T} \sum_{t=1}^{T} \tilde{s}_{y_t}
$$

This loss measures the quality of text generation by averaging the scores over the sequence length \(T\).

The embedding loss function, \(\mathcal{L}_{Emb}\), is given by:

$$
\mathcal{L}_{Emb}(x, y, y') = (1 - l) \cdot D(f(x), f(y))^2 + l \cdot \max\left(0, \alpha - D(f(x), f(y'))\right)^2
$$

This loss ensures that the embeddings are learned effectively by balancing the distance between the correct pairs \((x, y)\) and the incorrect pairs \((x, y')\).

The combined loss function, \(\mathcal{L}_{Mix}\), used for training the model is:

$$
\mathcal{L}_{Mix}=\lambda_{Emb}\mathcal{L}_{Emb}+\lambda_{Gen}\mathcal{L}_{Gen}
$$

This mixed loss function integrates both the embedding and generative tasks, where \(\lambda_{Emb}\) and \(\lambda_{Gen}\) are the respective weights for each loss component.

By using this mixed-task training approach, the model is capable of both text generation and embedding tasks effectively.

## Requirements
```
accelerate>=0.26.1
transformers>=4.37.2
datasets>=2.16.1
wandb
mteb[beir]
```

## Usage 


```python
from egtlm import EgtLM
from tqdm import tqdm
from scipy.spatial.distance import cosine

model = EgtLM(
    "selmisskilig/EGTLM-Qwen1.5-1.8B-instruct",
    mode="unified",
    torch_dtype="auto",
    attn_implementation="eager"
)

messages_list = [
    [{"role": "user", "content": "请帮我写一首李白的诗"}],
    [{"role": "user", "content": "多少岁才能够算成年？"}],
    [{"role": "user", "content": "请帮我写一个睡前小故事，来安慰我的宝宝睡觉。"}],
    [{"role": "user", "content": "请问中国有多少个朝代？"}],
]


def egtlm_instruction(instruction):
    return (
        "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    )


for messages in tqdm(messages_list):
    print("Query:\n", messages[0]["content"])
    
    encoded = model.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    encoded = encoded.to(model.device)
    gen = model.model.generate(encoded, max_new_tokens=256, do_sample=False)
    decoded = model.tokenizer.batch_decode(gen)
    print("Answer:\n")
    print(decoded[0], "\n====================\n")


queries = ["请告诉我比特币是怎样运作的？", "请问美国有多少年的发展历史？"]
documents = [
    "纯粹的点对点电子现金可以让在线支付直接从一方发送到另一方，而无需通过金融机构。数字签名提供了部分解决方案，但如果仍然需要一个可信的第三方来防止双重消费，则会失去主要的好处。我们提出了一种利用点对点网络解决双重消费问题的方案。网络通过将交易散列到一个持续的基于散列的工作证明链中来为交易打上时间戳，这样就形成了一个记录，如果不重做工作证明，就无法更改该记录。最长的链不仅可以证明所见证的事件顺序，还可以证明它来自最大的 CPU 能力池。只要大部分 CPU 能力由不合作攻击网络的节点控制，它们就能生成最长的链，并超越攻击者。网络本身的结构要求极低。信息在尽最大努力的基础上进行广播，节点可以随意离开和重新加入网络，并接受最长的工作证明链作为它们离开时发生的事情的证明。",
    """美国作为一个独立国家的历史可以追溯到1776年7月4日,当时美国十三个殖民地通过《独立宣言》正式脱离英国统治,宣布独立。因此,从1776年独立宣言签署算起,到2023年,美利坚合众国已经有247年的历史。不过,如果从欧洲人最早在北美洲定居开始算起,美国的历史可以追溯到1607年,当时英国人在今日维尔jinnia州的詹姆斯敦建立了第一个永久性英国殖民地。从1607年算起,到2023年,美国的历史已经超过415年了。当然,在欧洲人到来之前,北美洲大陆上已经有众多印第安人部落生活了数千年。所以从更广阔的视角来看,美国这片土地上的人类历史可以追溯到更加悠久的时期。总的来说,作为一个国家,美国有247年的独立历史;作为一片土地上的人类文明,美国的历史可以追溯到早于欧洲人到来的印第安人时期,时间跨度超过數千年。""",
]

d_rep, d_cache = model.encode(
    documents, instruction=egtlm_instruction(""), get_cache=True
)
q_rep = model.encode(queries, instruction=egtlm_instruction(""))

sims = {
    q: [1 - cosine(q_rep[i], d_rep[j]) for j in range(len(d_rep))]
    for i, q in enumerate(queries)
}

print(sims)
```

## Citation

```bibtex

@misc{muennighoff2024generative,
      title={Generative Representational Instruction Tuning}, 
      author={Niklas Muennighoff and Hongjin Su and Liang Wang and Nan Yang and Furu Wei and Tao Yu and Amanpreet Singh and Douwe Kiela},
      year={2024},
      eprint={2402.09906},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
