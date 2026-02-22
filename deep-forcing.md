# Deep Forcing: Training-Free Long Video Generationwith Deep Sink and Participative Compression

Jung Yi Wooseok Jang Paul Hyunbin Cho Jisu Nam Heeji Yoon Seungryong Kim

KAIST AI

https://cvlab-kaist.github.io/DeepForcing

![](images/28fda05137ab5ca71bc553fd8d154636cefc6517b36697e25bc779f085acd716.jpg)


![](images/7cff86fc777bfcc7c98ec9eebc82ba509b1b383169874031cebd31db7b80d350.jpg)



Figure 1. Our training-free approach, Deep Forcing, achieves comparable visual quality to training-based baselines, such as RollingForcing [17] and LongLive [31]. Notably, Deep Forcing enables minute-long video generation while maintaining visual quality anddynamics without requiring any additional training.


# Abstract

Recent advances in autoregressive video diffusion have en-abled real-time frame streaming, yet existing solutions stillsuffer from temporal repetition, drift, and motion deceler-ation. We find that na¨ıvely applying StreamingLLM-styleattention sinks to video diffusion leads to fidelity degrada-tion and motion stagnation. To overcome this, we introduceDeep Forcing, which consists of two training-free mech-anisms that address this without any fine-tuning. Specif-

ically, 1) Deep Sink dedicates half of the sliding windowto persistent sink tokens and re-aligns their temporal RoPEphase to the current timeline, stabilizing global context dur-ing long rollouts. 2) Participative Compression performsimportance-aware KV cache pruning that preserves only to-kens actively participating in recent attention while safelydiscarding redundant and degraded history, minimizing er-ror accumulation under out-of-distribution length gener-ation. Together, these components enable over $1 2 \times$ ex-trapolation (e.g. 5s-trained $ 6 0 s +$ generation) with bet-

ter imaging quality than LongLive, better aesthetic qual-ity than RollingForcing, almost maintaining overall consis-tency, and substantial gains in dynamic degree, all whilemaintaining real-time generation. Our results demon-strate that training-free KV-cache management can matchor exceed training-based approaches for autoregressivelystreaming long-video generation.

# 1. Introduction

Recent advances in video diffusion models [10, 25, 33] havedemonstrated remarkable capabilities in synthesizing shortvideo clips (e.g, 50–81 frames) with both high visual fi-delity and coherent motion dynamics.

Building upon this progress, emerging interactive sys-tems, such as world models [1, 18, 34], now require au-toregressive video generation (e.g., 1-2 minutes), whereframes are streamed sequentially in real time for imme-diate downstream applications [18, 21, 34]. Unlike con-ventional offline video generation, which synthesizes entireclips at once, autoregressive video generation operates inan online manner, with each frame generated, emitted, andconsumed instantaneously. Self Forcing [12] and its vari-ants [8, 17, 31] have become standard in this field, leverag-ing a causal attention mask and the Key-Value (KV) cachefrom previous frames.

However, this existing autoregressive formulation is in-herently susceptible to error accumulation over long hori-zons, as each predicted frame depends on previously gen-erated and potentially imperfect frames [12, 27, 35]. Suchaccumulation error leads to fidelity degradation, in whichvisual quality deteriorates as colors drift toward over-saturation, textures blur, and fine details disappear.

To mitigate this, several works [4, 22] introduce historycorruption by adding noise to previous frames during train-ing. While this improves robustness to noisy generated his-tories at inference, a discrepancy remains between gener-ated noise and artificially injected noise, leaving modelsvulnerable to long-horizon drift. Self Forcing [12] aims toreduce this gap by training on self-generated histories, buttheir heavy reliance on past frames’ KV caches still leads toaccumulated errors.

On the other hand, recent LLM studies [9, 16, 20], in-spired by StreamingLLM [29], introduce an attention sink,in which newly generated tokens attend strongly to a smallset of initial global tokens, helping stabilize the attentiondistribution and improve overall performance.

Motivated by these insights, we propose Deep Forc-ing, a novel tuning-free method that addresses error ac-cumulation in long-horizon video generation. Our ap-proach is able to generate minute-long video that main-tain both visual fidelity and motion stability, while requir-ing no fine-tuning, outperforming even training-based ap-

![](images/b3ebdc4808a57125be9648a75af0471ba73d60fbd26f7cd623ce527dfa2f79fa.jpg)



Figure 2. Comparison of KV Cache Management. (a) SelfForcing [12] adopts a FIFO policy that discards the earliest to-kens regardless of their importance, often losing critical contextand degrading generation quality. In contrast, our (b) Deep Forc-ing performs selective eviction by preserving Deep Sink tokensand applying KV-cache compression, effectively mitigating visualdegradation during long-horizon generation.


proaches [17, 31] (Fig. 1).

Specifically, we observe that the pre-trained Self Forc-ing [12] inherently exhibits strong attention sink behavior,not only attending to the first few tokens but also strongly at-tending to intermediate tokens. Building on this finding, wefirst introduce Deep Sink, which (i) maintains a large deep-sink ratio (typically $4 0 \mathrm { - } 6 0 \%$ ) and (ii) dynamically adjuststhe Relative Positional Embedding (RoPE) for long videogeneration. Second, we further present Participative Com-pression that retains only the most informative tokens in theKV cache by selecting them based on their importance toqueries from recent frames. This removes redundancy andsignificantly reduces fidelity degradation caused by noiseaccumulation from outdated tokens.

We implement our method on top of the pre-trained SelfForcing. Comprehensive evaluations using VBench [13],user studies, and VLM assessments demonstrate that ourtraining-free approach significantly enhances the baselinewithout any fine-tuning. We also achieve state-of-the-artperformance on several metrics, even surpassing training-based methods [17, 31]. Our ablation studies further vali-date the effectiveness of each design choice.

Our contribution is summarized as follows:

• We propose a tuning-free autoregressive video genera-tion framework, dubbed Deep Forcing, that significantlymitigates error accumulation in long-horizon generation.

• We introduce Deep Sink, which stabilizes long-horizongeneration by leveraging the inherent attention sink be-havior of Self Forcing [12] while adjusting the relativepositional gap.

• We present Participative Compression, a lightweightKV selection mechanism that removes redundant tokens.

• Our training-free method achieves state-of-the-art perfor-mance on VBench and user studies, surpassing existingtraining-based approaches.


(a) Deep Forcing


![](images/bc6567f4d25e9027d5814de5b0cff055839732870fa9f62df8486d43748ad96e.jpg)



(b) Participative Compression


![](images/a3c39e09d9c38a06e3a87cd012d04d37739de7dfc4f2b1574ca39b879854e19a.jpg)



Figure 3. Overview of Deep Forcing. (a) Deep Forcing maintains a substantially enlarged attention sink (Deep Sink) covering approxi-mately half the context window, combined with Participative Compression for the remaining rolling portion. Temporal RoPE adjustmentaligns the sink tokens’ temporal indices with current frames to maintain temporal coherence. (b) Participative Compression computesquery-averaged attention scores between recent tokens and candidate tokens, selecting the top-C most important tokens to retain in thecompressed cache while evicting redundant tokens.


# 2. Related Work

Autoregressive Video Diffusion. A growing line ofwork [4, 12, 14, 24, 35] combines diffusion modeling withautoregressive (AR) prediction to support long-horizon orstreaming video generation. MAGI-1 [24] generates videoschunk-by-chunk autoregressively with progressive denois-ing, enabling streaming generation. CausVid [35] con-verts a pre-trained bidirectional diffusion transformer intoa causal AR generator with KV caching. Building onthese ideas, Self Forcing [12] addresses the train–inferencemismatch by conditioning the model on its own generatedframes. Rolling Forcing [17] proposes expanding the diffu-sion window, and LongLive [31] incorporates KV recachingto maintain visual continuity while ensuring prompt adher-ence across scene transitions. In contrast, our method isfully tuning-free. We show that the pre-trained Self Forcingalready has attention-sink behavior and demonstrate howto leverage it effectively to surpass existing training-basedmethods.

Attention Sink. Recent work has revealed that atten-tion in autoregressive models concentrates dispropor-tionately on initial tokens, termed attention sinks [29].StreamingLLM [29] showed that retaining these sink tokenswithin a sliding window enables robust generation beyondthe training context length.

Building on this insight, recent autoregressive videomodels [17, 31] maintain the first three frames as atten-tion sinks via model distillation or fine-tuning. We demon-strate that the pre-trained autoregressive video diffusionmodel [12] exhibits inherent attention sink behavior that canbe effectively leveraged without training, requiring deepercontext preservation.

KV Cache Compression. The linearly growing KVcache in autoregressive generation motivates compressionstrategies that reduce memory footprint while preservinggeneration quality. As the cache grows, attention becomesdistributed across increasingly many tokens, diluting focuson critical context and degrading output quality. To ad-dress this, recent works employ attention-based token se-lection for long-context LLM generation. H2O [37] andSnapKV [16] preserve important tokens based on cumu-lative attention scores and observation windows, respec-tively. D2O [26] dynamically allocates budgets across lay-ers, while MorphKV [9] maintains constant-size cachesthrough correlation-aware ranking. While these methodstarget language models, similar memory constraints arisein autoregressive video diffusion, where temporal contextmust be efficiently maintained across frames. We extendthese principles through Participative Compression.

# 3. Preliminaries

Autoregressive Video Diffusion. Autoregressive videodiffusion models [5, 12, 24] produce each frame or chunkconditioned on previously generated frames within a de-noising diffusion process.

Given a video sequence of $N$ frames $\begin{array} { r l } { x ^ { 1 : N } } & { { } = } \end{array}$$( x ^ { 1 } , x ^ { 2 } , \ldots , x ^ { N } )$ , the autoregressive model applies thechain rule to factorize the joint distribution as

$$
p \left(x ^ {1: N}\right) = \prod_ {i = 1} ^ {N} p \left(x ^ {i} \mid x ^ {<   i}\right). \tag {1}
$$

A diffusion model parameterizes each conditional $p ( x ^ { i } \mid$$x ^ { < i }$ ), generating the $i$ -th frame by conditioning on previ-ously generated frames $x ^ { < i } = ( x ^ { 1 } , x ^ { 2 } , \ldots , x ^ { i - 1 } )$ .

Self Forcing. Self Forcing [12] generates videos in an au-L0 H10toregressive manner using a rolling KV cache mechanism,producing frames or frame chunks sequentially, enabling ef-ficient long video generation. Each frame is encoded intolatent tokens through the VAE encoder. The method main-tains a fixed-size cache of length $L$ that stores key-valuepairs corresponding to the most recent $L$ frames. When thecache reaches capacity, the oldest entry is evicted to accom-modate new frames, thereby maintaining a sliding contextwindow over the L5 H10 $L$ most recent frames. During genera-tion, self-attention is computed between queries from theframe(s) currently being generated and the keys and val-ues of cached context frames. Specifically, Self Forcingemploys a 4-step denoising process with noise schedule$\{ t _ { 0 } ~ = ~ 0 , t _ { 1 } ~ = ~ 2 5 0 , t _ { 2 } ~ = ~ 5 0 0 , t _ { 3 } ~ = ~ 7 5 0 , t _ { 4 } ~ = ~ 1 , 0 0 0 \}$$ { T ^ { \mathrm { ~ ~ } } } =  { 4 } )$ , totaling 5 noise levels. Each frame $i$ is de-noised iteratively across these timesteps. At denoising step6 8 10 12 14 16 18 20$j$ , the model processes a noisy intermediate frame Key Frame Index $\boldsymbol { x } _ { t _ { j } } ^ { i }$ , con-ditioned on the KV cache of previously generated cleanframes. The predicted clean frame is then perturbed withGaussian noise at a lower noise level $t _ { j - 1 }$ via the forwarddiffusion process $\Psi$ , producing $\boldsymbol { x } _ { t _ { j - 1 } } ^ { i }$ for the next denoisingiteration. Formally, this process is expressed as:

$$
x _ {t _ {j - 1}} ^ {i} = \Psi \left(G _ {\theta} \left(x _ {t _ {j}} ^ {i}, t _ {j}, K V\right), t _ {j - 1}\right), \tag {2}
$$

where $x _ { t _ { 4 } } ^ { i } \sim \mathcal { N } ( 0 , I )$ and $K V$ denotes the KV cache frompreviously generated frames.

# 4. Method

# 4.1. Overview

We propose a novel training-free method to mitigate erroraccumulation in long-horizon video generation. Drawinginspiration from the attention sink mechanism in large lan-guage models (LLMs) [29], our work begins by thoroughlyinvestigating the attention mechanism within the pre-trainedSelf Forcing [12]. Based on this investigation, we introducetwo core components: Deep Sink, which maintains approx-imately half of the sliding window as attention sinks, andParticipative Compression, which selectively retains im-portant tokens in the KV cache, while evicting redundantones. Our method is illustrated in Fig. 3

# 4.2. Deep Sink

Motivation. Self Forcing [12] employs a sliding windowto autoregressively extrapolate video frames. However, be-cause the model is distilled from short video clips (e.g.,5-second segments), its frame fidelity deteriorates signifi-cantly when generating sequences that extend far beyondits training domain.

This degradation is a known challenge in autoregressivesystems. In the LLM domain, the attention sink mecha-nism [29] was introduced as a simple yet effective tech-

![](images/5da3fd40072f60007736113fcb8ff75bf0b59e407a945365e06ae49ca7569688.jpg)


![](images/f59ac06db30e9453bd0c5d15b6a0a91495f6fc18b978b44579d953c396fd1416.jpg)



Figure 4. Attention weight distribution across earlier frames.Query-averaged attention showing how the last chunk (frames 19-21) attends to earlier KV cache entries (frames 0-18). We visualizetwo representative attention heads from different layers—L1H1(layer 1, head 1) and L5H10 (layer 5, head 10)—demonstratingthat substantial attention is maintained across the entire contextwindow, not just initial frames. See Appendix H for additionalheads analysis.


nique to mitigate performance drift during sliding-windowinference. While several works [17, 31] have investigatedadapting the attention sink mechanism to redistribute atten-tion probabilities and stabilize LLM performance, no priorwork has explored how to achieve a similar stabilizing effectin autoregressive video diffusion models in a training-freemanner.

To address this gap, we first analyze the attention behav-ior of the pre-trained Self Forcing. As illustrated in Fig. 4,we specifically examine how newly generated latent framesattend to earlier frames in the KV cache. Contrary to theconventional understanding [29] that only a small set ofinitial KV tokens (latent frames) needs to be retained, ouranalysis reveals: most attention heads allocate substantialweight to not only the earliest tokens, but also assign com-parable attention to the middle of the sequence.

Deepening Sink Size. Based on this observation, we hy-pothesize that more tokens up to the middle of the initialsequence are essential for high-quality video generation.To evaluate this hypothesis, we measured the influence ofdifferent attention sink sizes on the generation quality oflong videos. To rigorously assess long-horizon generationquality, we first define our key metrics from VBench [13]:Overall Consistency and Aesthetic Quality, which use Vi-

![](images/1ae546939eb5fa8c1e3470cd0ac18360b31b5f275181661bb8b8d9eafeaade0f.jpg)



Figure 5. Ablation study on Deep Sink depth. We evaluate theeffect of sink depth on video quality using Aesthetic Drift (↓) andOverall Consistency (↑) metrics on 50-second videos from the first21 prompts in MovieGen [19].


CLIP [28] and LAION aesthetic predictor [15], respec-tion [17, 35, 36], we compute tively. Following standard practice in long video genera- $\Delta _ { \mathrm { D r i f t } } ^ { \mathrm { Q u a l i t y } }$ as the absolute dif-ference in aesthetic quality between the first and the lastfive seconds of each 50-second generated video. Our re-sults demonstrate a clear trend (Fig. 5): as the sink framesize increases,Quality Drift $( \Delta _ { \mathrm { D r i f t } } ^ { \mathrm { Q u a l i t y } } )$ Consistency improves and Aesthetic decreases. This finding suggeststhat intermediate frames function as crucial anchors, ef-fectively maintaining both temporal coherence and visualfidelity throughout the long generation process. Conse-quently, we find that effective attention sinking in Self Forc-ing [12] emerges from deep, extended temporal anchoring,a mechanism that differs from the shallow, initial-frame fix-ation used in StreamingLLM [29].

Temporal RoPE Adjustment. RoPE [23] is widely usedas the positional embedding in video diffusion models, andrecent architectures [6, 24, 25, 33] commonly adopt 3DRoPE, which encodes temporal, height, and width dimen-sions separately. However, attention sinks in video requirethe model to attend to past frames, and directly applying3D RoPE under this setting leads to large temporal discrep-ancies, where tokens at vastly different timestamps (e.g.,$t = 1$ vs. $t = 2 0 0$ ) are forced to attend to each other. Thisbreaks the continuity of video and results in (1) flickering,(2) fidelity degradation, and (3) roll-back, where previouslysinked frames are regenerated (a detailed analysis is pro-vided in the Appendix A). To address this, we propose se-lectively adjusting only the temporal dimensions while pre-serving the original spatial encoding.

Specifically, we selectively modify the temporal RoPEindex by applying a temporal offset to the attention sink’stemporal index. This reduces the temporal gap between theattention sink and the remaining tokens, while preservingthe spatial indices unchanged.

We divide the key and value caches $K$ and $V$ in the cur-

rent sliding window into two parts: the sink $( K _ { \mathrm { s i n k } } , V _ { \mathrm { s i n k } } )$ fordeep sink tokens and the tail $( K _ { \mathrm { t a i l } } , V _ { \mathrm { t a i l } } )$ for the rest.

$$
K = \left[ K _ {\text {s i n k}} \| K _ {\text {t a i l}} \right], \tag {3}
$$

$$
V = \left[ V _ {\text {s i n k}} \| V _ {\text {t a i l}} \right], \tag {4}
$$

where $[ \cdot \parallel \cdot ]$ denotes concatenation.

Let $s _ { \mathrm { t a i l } }$ denote the first frame index of the tail and let$s _ { \mathrm { s i n k } }$ be the last frame index of the deep sink.

We then define $\Delta _ { \mathrm { s i n k } }$ , which is the temporal gap between$s _ { \mathrm { t a i l } }$ and $s _ { \mathrm { s i n k } }$ , as follows:

$$
\Delta_ {\text {s i n k}} = s _ {\text {t a i l}} - s _ {\text {s i n k}}. \tag {5}
$$

We apply $\Delta _ { \mathrm { s i n k } }$ to $K _ { \mathrm { s i n k } } ^ { ( \mathrm { t i m e ) } }$ , which is temporal componentof $K _ { \mathrm { s i n k } }$ , using the RoPE temporal frequency vector $\omega _ { t }$ :

$$
K _ {\text {s i n k}} ^ {(\text {t i m e})} \leftarrow K _ {\text {s i n k}} ^ {(\text {t i m e})} \odot \exp \left(i \omega_ {t} \Delta_ {\text {s i n k}}\right), \tag {6}
$$

where $i$ is the imaginary unit, $\odot$ denotes element-wise mul-tiplication. This further rotates $K _ { \mathrm { s i n k } }$ to align the relativetemporal positions of sink and tail tokens.

# 4.3. Participative Compression

Motivation. While Deep Sink effectively mitigates fi-delity degradation compared to the baseline Self Forc-ing [12], it alone cannot fully alleviate quality degrada-tion in minute-long video generation. When extrapolatingfrom 5-second training clips to sequences more than $1 2 \times$longer, a critical issue emerges: degeneration, where visualfidelity and overall quality progressively deteriorate. Thisphenomenon is well-documented in autoregressive long-context generation [9, 11]: when generating beyond train-ing length, indiscriminate token retention causes attentionto dilute across both relevant and irrelevant context, intro-ducing compounding noise. Beyond its training distribu-tion, the growing KV cache retains increasingly irrelevanttokens, further diluting attention.

Recent analysis of video diffusion models reveals that at-tention concentrates on a small subset of semantically criti-cal tokens, with the majority contributing minimally to gen-eration [32]—suggesting that pruning low-attention tokenscan substantially reduce computation with limited impacton quality. Building on this insight and importance-awarecompression [9, 16, 37], we propose Participative Com-pression, which dynamically identifies and retains onlycontextually relevant tokens while pruning those that couldcontribute to attention dilution and error accumulation.

Overview. Self Forcing [12] implements the rolling KVcache by evicting the earliest frame when the cache is filled.In comparison, Participative Compression (PC) operates atthe token level, selectively removing redundant tokens by

ranking them according to their aggregated attention scoresfrom recent frames, rather than using a simple FIFO (First-In, First-Out) policy as illustrated in Fig 2.

PC introduces two key hyperparameters: (1) Budget$( N )$ , the target number of tokens to retain after compression,and (2) Recent $( R )$ , the number of tokens from the mostrecent frames that are excluded from compression, in addi-tion to the $S$ sink frame tokens that are always preserved.PC is applied when the sliding window reaches maximumlength $M$ tokens, compressing the cache to size $N \leq M$ .The compression operates on $K _ { \mathrm { c a n d } }$ , $V _ { \mathrm { c a n d } }$ , which contain alltokens except those from the first $S$ and the most recent $R$ .

• Recent: $K _ { \mathrm { r c t } } , V _ { \mathrm { r c t } }$ containing tokens from the most recent$R$ frames, excluded from compression to preserve localcoherence.

• Candidate: $K _ { \mathrm { c a n d } } , V _ { \mathrm { c a n d } }$ containing all intermediate to-kens between the Sink and Recent, subject to compres-sion.

For each token in $K _ { \mathrm { c a n d } } , V _ { \mathrm { c a n d } }$ , PC computes an impor-tance score by summing its attention weights from all recent$R$ frames—tokens frequently attended to are deemed criti-cal for maintaining temporal coherence. PC then selects thetop $C = N - R - S$ tokens with the highest importancescores to form $K _ { \mathrm { t o p } } , V _ { \mathrm { t o p } }$ . The final KV cache contains $N$tokens: $S$ sink tokens, $C$ compressed tokens, and $R$ recenttokens.

Top- $C$ Selection. PC selectively retains the $C$ most im-portant candidate tokens based on their relevance to cur-rent generation, evicting those not selected by the Top-Coperator. To determine which tokens to retain, PC com-putes attention scores between the recent queries $( Q _ { \mathrm { r c t } } )$ andcandidate keys $( K _ { \mathrm { c a n d } } )$ . We aggregate these scores acrossall recent queries by summing along the query dimension,producing a unified importance score $\phi _ { j }$ for each candidatekey:

$$
\phi_ {j} = \sum_ {r = 1} ^ {R} \mathbf {q} _ {r} ^ {\top} \mathbf {k} _ {j}, \tag {7}
$$

where $j$ indexes the candidate keys, $\mathbf { q } _ { r }$ denotes the $r$ -thquery in $Q _ { \mathrm { { r c t } } }$ , and $\mathbf { k } _ { j }$ denotes the $j$ -th key in $K _ { \mathrm { c a n d } }$ . Higher$\phi _ { j }$ indicates higher importance for current generation. Wethen form the importance vector $\phi = \left[ \phi _ { 1 } , \phi _ { 2 } , \ldots , \phi _ { \left| K _ { \mathrm { c a n d } } \right| } \right]$and select the Top- $C$ tokens with the highest scores:

$$
K _ {\text {t o p}} = \operatorname {T o p} - \mathrm {C} (\phi). \tag {8}
$$

Finally, the compressed cache is formed by concatenat-ing the preserved components in temporal order:

$$
K _ {\text {c o m p r e s s e d}} = \left[ K _ {\text {s i n k}} \| K _ {\text {t o p}} \| K _ {\text {r c t}} \right], \tag {9}
$$

where $K _ { \mathrm { r c t } }$ contain keys from the first $S$ and most recent $R$ ,respectively. Values $( V _ { \mathrm { t o p } } )$ are processed identically. This


Algorithm 1 Participative Compression with Deep Sink


Input: KV cache  $[K,V]$  of size  $M$ ; Sink size  $S$ ; Recent  $R$ ; Top-C capacity  $C$ ; Timestep  $t$ ; first time step  $T$ ;  
1: if  $M \geq$  MAXWINDOW_LENGTH and  $t = T$  then  
2: // Partition cache into three regions  
3:  $\mathcal{I}_{\mathrm{sink}} \gets [0,S)$  ▷ First  $S$  frames  
4:  $\mathcal{I}_{\mathrm{rect}} \gets [M - R,M)$  ▷ Last  $R$  frames  
5:  $\mathcal{I}_{\mathrm{cand}} \gets [S,M - R)$  ▷ Candidate tokens/frames  
6: if  $|\mathcal{I}_{\mathrm{cand}}| > 0$  and  $C > 0$  then  
7: // Compute importance scores (Eq. 7)  
8:  $Q_{\mathrm{rect}} \gets Q[\mathcal{I}_{\mathrm{rect}}]$  ▷ Recent queries  
9:  $K_{\mathrm{cand}} \gets K[\mathcal{I}_{\mathrm{cand}}]$  ▷ Candidate keys  
10: for  $j = 1$  to  $|\mathcal{I}_{\mathrm{cand}}|$  do  
11:  $\phi_j \gets \sum_{r=1}^{R} \mathbf{q}_r^{\top} \mathbf{k}_j$  ▷ Aggregate attention  
12: // Select top-C tokens (Eq. 8)  
13:  $\phi \gets [\phi_1, \phi_2, \dots, \phi_{|\mathcal{I}_{\mathrm{cand}}|}]$   
14:  $\mathcal{I}_{\mathrm{top}} \gets \mathrm{TOPC}(\phi)$  ▷ Select  $C$  highest  
15: // Temporal RoPE Unification (Section 4.3)  
16:  $\Delta_{\mathrm{top}} \gets s^{\mathrm{top}} - s_{\mathrm{base}}^{\mathrm{top}}$   
17:  $K_{\mathrm{top}}^{\mathrm{(time)}} \gets K_{\mathrm{top}}^{\mathrm{(time)}} \odot \exp(i\omega_t \Delta_{\mathrm{top}})$   
18: else  
19:  $\mathcal{I}_{\mathrm{top}} \gets \emptyset$   
20: // Assemble compressed cache (Eq. 9)  
21:  $K_{\mathrm{compressed}} \gets [K_{\mathrm{sink}} \| K_{\mathrm{top}} \| K_{\mathrm{rect}}]$   
22:  $V_{\mathrm{compressed}} \gets [V_{\mathrm{sink}} \| V_{\mathrm{top}} \| V_{\mathrm{rect}}]$   
23: return  $K_{\mathrm{compressed}}, V_{\mathrm{compressed}}$   
24: else  
25: return  $K, V$  ▷ No compression

yields a compact cache structure combining long-term ini-tial context (Sink), selectively important intermediate to-kens (Top- $C$ ), and fresh recent context (Recent), all withina fixed budget of $N$ .

Temporal RoPE Unification. After selecting the Top- $C$tokens, we apply RoPE adjustment to maintain temporal di-mension consistency, following the same approach as DeepSink (Section 4.2). We adjust only the temporal dimensionof the Top- $C$ keys’ RoPE while preserving their spatial in-formation intact.

Let $s ^ { \mathrm { t o p } }$ denote the desired absolute temporal positionwhere the Top- $C$ block should be aligned, and let $s _ { \mathrm { b a s e } } ^ { \mathrm { \bar { t o p } } }$ rep-resent the current temporal position of each cached Top- $C$key. We compute the temporal adjustment:

$$
\Delta_ {\text {t o p}} = s ^ {\text {t o p}} - s _ {\text {b a s e}} ^ {\text {t o p}}. \tag {10}
$$

This temporal shift is temporal componen $\Delta _ { \mathrm { t o p } }$ hen applied to , re-aligning e $K _ { \mathrm { t o p } } ^ { \mathrm { ( t i m e ) } }$ hichkey$K _ { \mathrm { t o p } }$ $C$


Table 1. Quantitative comparison on long video generation. We evaluate Deep Forcing against open-source autoregressive video diffusiongeneration baselines on 30-second and 60-second videos across multiple quality metrics on VBench-Long [13].


<table><tr><td>Model</td><td>Throughput (FPS) ↑</td><td>Dynamic Degree ↑</td><td>Motion Smoothness ↑</td><td>Overall Consistency ↑</td><td>Imaging Quality ↑</td><td>Aesthetic Quality ↑</td><td>Subject Consistency ↑</td><td>Background Consistency ↑</td></tr><tr><td>Trained with Attention Sink</td><td colspan="8">30 seconds</td></tr><tr><td>Rolling Forcing [17]</td><td>15.79</td><td>30.71</td><td>98.75</td><td>20.99</td><td>70.58</td><td>60.24</td><td>98.12</td><td>96.91</td></tr><tr><td>LongLive [31]</td><td>18.16</td><td>45.55</td><td>98.76</td><td>20.16</td><td>69.07</td><td>61.51</td><td>97.97</td><td>96.83</td></tr><tr><td>Trained without Attention Sink</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CausVid [35]</td><td>15.78</td><td>47.21</td><td>98.08</td><td>19.15</td><td>66.36</td><td>59.77</td><td>97.92</td><td>96.77</td></tr><tr><td>Self Forcing [12]</td><td>15.78</td><td>36.62</td><td>98.63</td><td>20.50</td><td>68.58</td><td>59.44</td><td>97.34</td><td>96.47</td></tr><tr><td>Deep Forcing (Ours)</td><td>15.75</td><td>57.56</td><td>98.27</td><td>20.54</td><td>69.31</td><td>60.68</td><td>97.34</td><td>96.48</td></tr><tr><td>Trained with Attention Sink</td><td colspan="8">60 seconds</td></tr><tr><td>Rolling Forcing [17]</td><td>15.79</td><td>31.35</td><td>98.69</td><td>20.64</td><td>70.25</td><td>59.75</td><td>97.97</td><td>96.76</td></tr><tr><td>LongLive [31]</td><td>18.16</td><td>43.49</td><td>98.75</td><td>20.29</td><td>69.11</td><td>61.29</td><td>97.85</td><td>96.74</td></tr><tr><td>Trained without Attention Sink</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CausVid [35]</td><td>15.78</td><td>46.44</td><td>98.09</td><td>18.78</td><td>65.84</td><td>59.42</td><td>97.81</td><td>96.75</td></tr><tr><td>Self Forcing [12]</td><td>15.78</td><td>31.98</td><td>98.21</td><td>18.63</td><td>66.33</td><td>56.45</td><td>96.82</td><td>96.31</td></tr><tr><td>Deep Forcing (Ours)</td><td>15.75</td><td>57.19</td><td>98.23</td><td>20.38</td><td>69.27</td><td>59.86</td><td>96.96</td><td>96.32</td></tr></table>

using the complex phase rotation defined by the RoPE tem-poral frequencies $\omega _ { t }$ :

$$
K _ {\text {t o p}} ^ {\left(\text {t i m e}\right)} \leftarrow K _ {\text {t o p}} ^ {\left(\text {t i m e}\right)} \odot \exp \left(i \omega_ {t} \Delta_ {\text {t o p}}\right). \tag {11}
$$

where $i$ is the imaginary unit, and $\odot$ denotes element-wisemultiplication.

This rotation adjusts the temporal positioning of $K _ { \mathrm { t o p } }$to create a continuous temporal sequence across all threecache components (Sink, Top- $C$ , Recent), preventing tem-poral discontinuities that would otherwise cause fidelitydegradation, flickering, and roll-back artifacts as demon-strated in Appendix A.

Efficiency. The complexity of Participative Compression(PC) might initially suggest a significant computationaloverhead. However, its computational burden is minimizeddue to its sparse activation criteria. PC is only engaged un-der two specific conditions: When the sliding context win-dow is completely filled, and during the first diffusion timestep $\left( t = T \right.$ ). Even though the Top- $C$ selection mechanisminvolves gathering and sorting tokens, our efficiency analy-sis justifies this operation in the Appendix E.

# 5. Experiments

# 5.1. Experimental settings

Implementation details. We implement chunk-wise SelfForcing [12] as our base model. We evaluate both quanti-tatively and qualitatively using Deep Sink (DS) combinedwith Participative Compression (PC). We set the hyperpa-rameters for Deep Sink and Participative Compression asfollows: sink size $S ~ = ~ 1 0$ , budget $N ~ = ~ 1 6$ , and recent$R = 4$ . We compare against baseline autoregressive videodiffusion models including CausVid [35], Self Forcing [12],Rolling Forcing [17], and LongLive [31].

Evaluation. We conduct evaluations under two settings.First, we evaluate long video generation on VBench-Long [13] using 128 prompts from MovieGen [19], fol-lowing the same prompt selection protocol as Self Forc-$\mathrm { i n g + + }$ [8]. Each prompt is refined using Qwen/Qwen2.5-7B-Instruct [30] following Self Forcing [12]. Second, weconduct a user preference study to evaluate color consis-tency, dynamic motion, subject consistency, and overallquality. Additional implementation details are provided inthe Appendix G. Third, we evaluate visual stability usingthe state-of-the-art vision-language model (VLM) Gemini2.5-Pro [7], following the protocol of Self Forcing $^ { + + }$ [8].

# 5.2. Results in Long Video Generation

Quantitative results. Our quantitative results are pre-sented in Table 1. Despite being a training-free methodbuilt upon Self Forcing, which was not trained for longvideo generation, our approach achieves performance com-parable to methods explicitly distilled or trained for longvideos [17, 31]. As shown in Table 1, we achieve supe-rior performance in overall consistency and imaging qual-ity compared to LongLive [31], and better aesthetic qualitythan Rolling Forcing [17].

Notably, our method also excels in dynamic degree, pro-ducing more dynamic motions than trained methods [17,31], despite not being explicitly optimized for this aspect.We attribute this to our training-free approach, which avoidsthe potential motion constraints introduced when modelsare explicitly trained to anchor with attention sinks.

Qualitative results. The qualitative results in Figure 8demonstrate strong visual quality, with our training-freemethod producing high-fidelity frames comparable to orbetter than training-based baselines. The results visuallyconfirm our quantitative performance, where we achieve

![](images/40736a0b0b339c85c2adace03a8b723f9a37db7dfa21f1194c71e3cff59625b0.jpg)



Figure 6. Qualitative ablation results over 30-second generation: Comparison of Self Forcing (SF) [12], SF with Deep Sink $( \mathrm { S F + D S }$ ),and SF with both Deep Sink and Participative Compression (Deep Forcing). Baseline SF exhibits severe color drift. $\mathrm { S F + D S }$ improvesstability but shows residual artifacts. Deep Forcing maintains consistent visual quality.


competitive scores without any fine-tuning. Notably, ourvideos exhibit more dynamic motion in both camera andsubject movement, yielding more visually expressive resultscompared to existing approaches.

Although subject consistency is lower in VBench-Longmetrics, the bottom example in Figure 8 demonstrates thatour training-free approach maintains better overall qualitywith limited degradation compared to training-based meth-ods. Additional qualitative results are provided in Ap-pendix F.

User study. To further validate these observations, weconducted a user study with 24 participants evaluating mul-tiple aspects of the generated videos. The user study wasconducted following the Two-Alternative Forced Choice(2AFC) protocol, where users are instructed to choosewhich is better between two videos (from Deep Forcingand a baseline), in regard to color consistency, dynamic mo-tion, subject consistency, and overall quality. As shown inTable 2, participants demonstrated a clear preference forDeep Forcing over the baselines across all evaluated as-pects. This includes a high preference in terms of subjectconsistency, highlighting Deep Forcing’s ability to retainthe subject with minimal identity drift throughout the video.These corroborate our qualitative assessment that percep-tual quality remains high despite lower VBench-Long [13]subject consistency scores.

VLM evaluation. For further comparison with the base-lines, we evaluate visual stability using the state-of-the-artvision–language model (VLM) Gemini 2.5-Pro [7]. Fol-lowing the protocol of Self Forcing $^ { + + }$ [8], we use the same


Table 2. User study results. Values represent percentage of votesfavoring our Deep Forcing over the baselines.


<table><tr><td>Method</td><td>Color Cons.</td><td>Dyn. Motion</td><td>Subject Cons.</td><td>Overall Quality</td></tr><tr><td>CausVid</td><td>98.9%</td><td>95.8%</td><td>96.8%</td><td>100%</td></tr><tr><td>Self Forcing</td><td>85.9%</td><td>86.9%</td><td>84.8%</td><td>87.9%</td></tr><tr><td>LongLive</td><td>71.2%</td><td>83.5%</td><td>72.2%</td><td>72.2%</td></tr><tr><td>Rolling Forcing</td><td>76.7%</td><td>76.7%</td><td>80.0%</td><td>78.9%</td></tr></table>


Table 3. Visual stability compared with the baselines. Methodsare additionally categorized by whether they are trained with anattention sink.


<table><tr><td>Method</td><td>Attention Sink Training</td><td>Visual Stability</td></tr><tr><td>CausVid [35]</td><td>No</td><td>42.84</td></tr><tr><td>Self Forcing [12]</td><td>No</td><td>43.94</td></tr><tr><td>Rolling Forcing [17]</td><td>Yes</td><td>72.6</td></tr><tr><td>LongLive [31]</td><td>Yes</td><td>78.58</td></tr><tr><td>Deep Forcing (Ours)</td><td>No</td><td>75.44</td></tr></table>

prompt to ask the VLM to score each 30-second video interms of exposure stability and degradation. Then we nor-malize the resulting scores 100. As shown in Tab. 3, ourtraining-free method achieves visual stability comparable tothat of recent methods [17, 31].

# 5.3. Ablation studies

We conducted ablation studies to evaluate the contributionsof each method. We measure relevant VBench-Long met-rics on 30 second videos.

Effect of Deep Sink & Participative Compression. Weevaluate three variants: naive Self-Forcing [12], Self-


Table 4. Ablation on our components: Deep Sink (DS) and Par-ticipative Compression (PC).


<table><tr><td>Method</td><td>Dynamic Degree</td><td>Overall Consistency</td><td>Image Quality</td></tr><tr><td>SF [12] (Baseline)</td><td>36.62</td><td>20.50</td><td>68.58</td></tr><tr><td>SF [12] + DS</td><td>48.58</td><td>20.54</td><td>68.54</td></tr><tr><td>SF [12] + DS + PC (Ours)</td><td>57.56</td><td>20.54</td><td>69.31</td></tr></table>

Forcing with only Deep Sink with sink length $S ~ = ~ 1 0$frames, and Self-Forcing with both Deep Sink $S = 1 0$frames) and Participative Compression $N = 1 6$ , $R = 4$ ). Asshown in Table 6, Deep Forcing demonstrates progressiveimprovements in dynamic degree, overall consistency, andimage quality as components are added. Notably, dynamicdegree improves substantially through the Deep Forcingframework, enabling the generation of significantly moredynamic scenes compared to baseline methods.

While image quality shows a slight decrease at 30 sec-onds, we have already demonstrated Deep Sink’s positiveimpact on 50-second videos in Section 4.2.

Ablation Visualization. Figure 6 visualizes our ablationstudy results. When generating long videos with Self Forc-ing (SF) alone (top row), error accumulation leads to se-vere fidelity degradation and visual quality deteriorates asHistorcolors drift toward over-saturation. Adding Deep Sink$\mathrm { ( S F + D S ) }$ substantially reduces fidelity degradation andmaintains more consistent colors. However, subtle arti-facts persist at frame 460, including slight color shift inthe coffee and texture blur in the ship details. When bothDeep Sink and Participative Compression are applied (DeepForcing), noticeable degradation is effectively eliminated.This validates that our complete framework successfullymitigates long-horizon error accumulation while preservingboth overall visual quality and fine-grained details.

Top-C Visualization. Figure 7 visualizes a subset of Top-$C$ tokens selected during the first rolling step, spatiallyaligned to their positions in Frame 37 when generatingFrame 82. The yellow highlighted regions indicate the spa-tial positions of tokens selected as Top- $C$ within the frame.These highlighted regions reveal semantic alignment withcontextually important content: the robot’s body and back-ground architecture, the octopus tentacles and crab, and thecircular coffee cup structure. This demonstrates that ourmethod identifies and retains semantically salient regionscritical for maintaining contextual coherence in subsequentgeneration.

# 6. Conclusion

We introduced Deep Forcing, a training-free approach forautoregressive long video generation that effectively mit-

![](images/ca180034598ef428a9855ba9bf412660732c01c0c805232c8dc3f9ddb9f60f8a.jpg)



A cyberpunk-style illustration depicting a lone robot navigating a neon-lit cityscape.The robot stands tall with sleek, metallic armor, adorned with blinking ,…


![](images/d2e4e9c08c7796e23e7a65d84673947f3a3d079fc4a733743f027890b5765ceb.jpg)



A wide-angle underwater photograph captures a large orange octopus resting on theocean floor, its tentacles spread out around its body and eyes closed…


![](images/248144715c76c6bbfe88d1e30299070e06abaeafd616e70bb0c8494f62715d0c.jpg)



A macro shot of a volcanic eruption in a coffee cup, capturing the dramatic moment invivid detail. The coffee cup is filled with rich,…



Figure 7. Visualization of Top- $C$ token selection. For each ex-ample, Frame 37 (middle) shows the Top- $C$ tokens selected forgenerating Frame 82 (right). Yellow highlights indicate the spa-tial locations of tokens chosen as Top- $C$ . Our method effectivelyidentifies and preserves regions that are critical for maintainingcontextual coherence during subsequent generation.


igates error accumulation through two key components:Deep Sink and Participative Compression. Our methodachieves state-of-the-art performance on VBench-Long,user studies, and VLM evaluation without any fine-tuning,even surpassing training-based methods. By exploiting theinherent deep attention sink behavior in pre-trained SelfForcing, we enable minute-long video generation while pre-serving both visual fidelity and motion dynamics. Thistraining-free paradigm offers a practical and efficient so-lution for long video generation with autoregressive videodiffusion.

Limitations and Future Works. While our training-freeapproach substantially improves long-horizon stability, sev-eral limitations remain. Operating at inference time on afrozen backbone, our method is constrained by the pre-trained model’s capacity and biases. Additionally, our ap-proach lacks explicit long-term memory, potentially causinggradual drift in extremely long sequences with repeated oc-clusions. Future work could integrate hierarchical memorymodules and extend to broader video generation settings.

![](images/6bf6f40f0c1ad057fc2b8b3ac499e48ed00643afc6163b52997c63e0ba9212d0.jpg)



Figure 8. Qualitative results on 30-second videos. Frame-by-frame comparison across different methods for two representative prompts.Deep Forcing (training-free) achieves temporal consistency and visual quality comparable to training-based baselines (CausVid [35], SelfForcing [12], LongLive [31], Rolling Forcing [17]) while generating more dynamic content with greater subject consistency.


# References



[1] Rtfm: A real-time frame modelhttps://www.worldlabs.ai/blog/rtfm, 2025. 2





[2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, DanielMendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi,Zion English, Vikram Voleti, Adam Letts, et al. Stable videodiffusion: Scaling latent video diffusion models to largedatasets. arXiv preprint arXiv:2311.15127, 2023. 5





[3] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, AdamPolyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin.Videojam: Joint appearance-motion representations for en-hanced motion generation in video models. arXiv preprintarXiv:2502.02492, 2025. 5





[4] Boyuan Chen, Diego Mart´ı Monso, Yilun Du, Max Sim- ´chowitz, Russ Tedrake, and Vincent Sitzmann. Diffusionforcing: Next-token prediction meets full-sequence diffu-sion. Advances in Neural Information Processing Systems,37:24081–24125, 2024. 2, 3





[5] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin,Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen,Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang,Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, YupengLiang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, De-bang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. Skyreels-v2: Infinite-length film generative model, 2025. 3





[6] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu,Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan,Daquan Zhou, Huan Ling, et al. Sana-video: Efficient videogeneration with block linear diffusion transformer. arXivpreprint arXiv:2509.24695, 2025. 5





[7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, IcePasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blis-tein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5:Pushing the frontier with advanced reasoning, multimodality,long context, and next generation agentic capabilities. arXivpreprint arXiv:2507.06261, 2025. 7, 8





[8] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, RuiWang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing $^ { + + }$ : Towards minute-scale high-quality video genera-tion. arXiv preprint arXiv:2510.02283, 2025. 2, 7, 8





[9] Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant J. Nair,and Poulami Das. Dialogue without limits: Constant-sized kv caches for extended responses in llms. ArXiv,abs/2503.00979, 2025. 2, 3, 5





[10] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, DanielShalem, Dudu Moshe, Eitan Richardson, Eran Levin, GuyShiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weiss-buch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, andOfir Bibi. Ltx-video: Realtime video latent diffusion. arXivpreprint arXiv:2501.00103, 2024. 2





[11] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and YejinChoi. The curious case of neural text degeneration. ArXiv,abs/1904.09751, 2019. 5





[12] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou,and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprintarXiv:2506.08009, 2025. 2, 3, 4, 5, 7, 8, 9, 10, 6





[13] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si,Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin,Nattapol Chanpaisit, et al. Vbench: Comprehensive bench-mark suite for video generative models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and PatternRecognition, pages 21807–21818, 2024. 2, 4, 7, 8





[14] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang,Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, andZhouchen Lin. Pyramidal flow matching for efficient videogenerative modeling. arXiv preprint arXiv:2410.05954,2024. 3





[15] LAION-AI. aesthetic-predictor. https://github.com/LAION- AI/aesthetic- predictor,. [Ac-cessed 14-11-2025]. 5





[16] Yuhong Li, Yingbing Huang, Bowen Yang, BharatVenkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, PatrickLewis, and Deming Chen. Snapkv: Llm knows whatyou are looking for before generation. arXiv preprintarXiv:2404.14469, 2024. 2, 3, 5





[17] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and ShijianLu. Rolling forcing: Autoregressive long video diffusion inreal time. arXiv preprint arXiv:2509.25161, 2025. 1, 2, 3, 4,5, 7, 8, 10, 6





[18] Jack Parker-Holder and Shlomi Fruchter. Genie 3: A newfrontier for world models, 2025. 2





[19] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra,Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast ofmedia foundation models. arXiv preprint arXiv:2410.13720,2024. 5, 7





[20] Dachuan Shi, Yonggan Fu, Xiangchi Yuan, ZhongzhiYu, Haoran You, Sixu Li, Xin Dong, Jan Kautz, PavloMolchanov, and Yingyan Celine Lin. Lacache: Ladder-shaped kv caching for efficient long-context modeling oflarge language models. In Forty-second International Con-ference on Machine Learning, 2025. 2





[21] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu,Jaesik Park, Eli Schechtman, and Xun Huang. Motion-stream: Real-time video generation with interactive motioncontrols. arXiv preprint arXiv:2511.01266, 2025. 2





[22] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du,Russ Tedrake, and Vincent Sitzmann. History-guided videodiffusion. arXiv preprint arXiv:2502.06764, 2025. 2





[23] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, WenBo, and Yunfeng Liu. Roformer: Enhanced transformer withrotary position embedding. Neurocomputing, 568:127063,2024. 5





[24] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li,Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang,Weifeng Luo, et al. Magi-1: Autoregressive video genera-tion at scale. arXiv preprint arXiv:2505.13211, 2025. 3, 5





[25] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao,Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianx-iao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jin-gren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao,Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang,Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei





Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui,Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang,Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu,Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, YangyuLv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yi-tong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, YunZheng, Yuntao Hong, Yupeng Shi, Yutong Feng, ZeyinziJiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Openand advanced large-scale video generative models. arXivpreprint arXiv:2503.20314, 2025. 2, 5





[26] Zhongwei Wan, Xinjian Wu, Yu Zhang, Yi Xin, Chao-fan Tao, Zhihong Zhu, Xin Wang, Siqi Luo, Jing Xiong,Longyue Wang, et al. D2o: Dynamic discriminative oper-ations for efficient long-context inference of large languagemodels. arXiv preprint arXiv:2406.13035, 2024. 3





[27] Jing Wang, Fengzhuo Zhang, Xiaoli Li, Vincent YF Tan,Tianyu Pang, Chao Du, Aixin Sun, and Zhuoran Yang. Erroranalyses of auto-regressive video diffusion models: A uni-fied framework. arXiv preprint arXiv:2503.10704, 2025. 2





[28] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu,Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, YaohuiWang, et al. Internvid: A large-scale video-text dataset formultimodal understanding and generation. arXiv preprintarXiv:2307.06942, 2023. 5





[29] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han,and Mike Lewis. Efficient streaming language models withattention sinks. arXiv preprint arXiv:2309.17453, 2023. 2,3, 4, 5





[30] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, ChengenHuang, Chenxu Lv, et al. Qwen3 technical report. arXivpreprint arXiv:2505.09388, 2025. 7





[31] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao,Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Ying-cong Chen, Yao Lu, et al. Longlive: Real-time interactivelong video generation. arXiv preprint arXiv:2509.22622,2025. 1, 2, 3, 4, 7, 8, 10, 5, 6





[32] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, JintaoZhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, KellyPeng, et al. Sparse videogen2: Accelerate video generationwith sparse attention via semantic-aware permutation. arXivpreprint arXiv:2505.18875, 2025. 5





[33] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, ShiyuHuang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiao-han Zhang, Guanyu Feng, et al. Cogvideox: Text-to-videodiffusion models with an expert transformer. arXiv preprintarXiv:2408.06072, 2024. 2, 5





[34] Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, JunZhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu Yang,Qiang Fu, et al. Yan: Foundational interactive video genera-tion. arXiv preprint arXiv:2508.08601, 2025. 2





[35] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Free-man, Fredo Durand, Eli Shechtman, and Xun Huang. Fromslow bidirectional to fast autoregressive video diffusion mod-els. In Proceedings of the Computer Vision and PatternRecognition Conference, pages 22963–22974, 2025. 2, 3,5, 7, 8, 10, 6





[36] Lvmin Zhang and Maneesh Agrawala. Packing input framecontext in next-frame prediction models for video genera-tion. arXiv preprint arXiv:2504.12626, 2025. 5





[37] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen,Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian,Christopher Re, Clark Barrett, et al. H2o: Heavy-hitter ora- ´cle for efficient generative inference of large language mod-els. Advances in Neural Information Processing Systems, 36:34661–34710, 2023. 3, 5



# Appendix

In this appendix, we provide comprehensive details, includ-ing:

• Comparison with other sink mechanisms [17, 31](Appendix A)

• Qualitative results on different sink sizes (Appendix B)

• Participative Compression details (Appendix C)

• Denoising query is not just noise (Appendix D)

• FPS measurements (Appendix E)

• More qualitative results (Appendix F)

• User study protocol (Appendix G)

• Additional attention visualization (Appendix H)

# A. The Tale of Three Sinks

Concurrent works [17, 31] propose different attention sinkmechanisms for Self Forcing-like architectures, typicallywith model training or distillation. In this section, we com-pare these approaches in a training-free setting to evaluatetheir effectiveness when applied directly to pretrained SelfForcing model.

Specifically, we compare three attention sink strategies:LongLive [31], Rolling Forcing [17], and Ours. LongLiveapplies attention sinks to the first 3 frames without RoPEadjustment. Rolling Forcing also uses 3 frame sinks butincorporates (1) storing raw keys and (2) dynamically re-applying RoPE when rolling occurs. Qualitative compar-isons are presented in Figure 9.

Figure 9 shows that our method achieves substantiallybetter results. The LongLive attention sink, which does notadjust RoPE, exhibits progressive failure modes: fidelitydegradation appears at frame 800, followed by flickeringartifacts at frame 801, and culminating in roll-back be-havior at frame 802 where the generation reverts to earlysinked frames. These issues also occur in LongLive [31],which was explicitly trained on long videos using this at-tention sink mechanism (Fig. 1).

Although Rolling Forcing attention sink [17] employsDynamic RoPE, which reapplies positional encodings to theentire cached key set, it still exhibits severe fidelity degra-dation at frames 800–801.

This comparison demonstrates that both deep sink andRoPE adjustment are essential for long video generation.

# B. Qualitative Results on Different Sink Size

Figure 10 presents qualitative comparisons across sink sizesranging from 0 to 18 frames, as analyzed in Section 4.2. Weevaluate two diverse prompts on 60-second generation.

With no attention sink (Sink 0), severe fidelity degrada-tion emerges rapidly, where the monster’s texture deterio-rates and colors shift noticeably by frame 230, with com-plete quality collapse by frame 690. Similarly, the SUVscene exhibits significant fidelity degradation. As the sink

size increases to 4 and 9 frames, degradation is progres-sively reduced but remains visible in fine details.

Once the sink size exceeds 10 frames (Sink 14), fidelitydegradation is substantially reduced. However, excessivelylarge sinks (Sink 18) exhibit repetitive generation whereearly frames are over-preserved.

These results validate our optimal sink range of 10–15frames $40 \%$ of the sliding window). While Deep Sinksubstantially mitigates degradation, it alone proves insuf-ficient to maintain visual fidelity throughout minute-longgeneration across diverse scenes, as demonstrated in our ex-tended evaluations (Section 5.3).

# C. Participative Compression Details

In this section, we provide additional details and analysis ofParticipative Compression (PC) beyond what was presentedin Section 4.3.

Each layer maintains its own KV cache, which under-goes compression as follows:

$$
\left[ K _ {\text {s i n k}} \| K _ {\text {c a n d}} \| K _ {\text {r c t}} \right]\rightarrow \left[ K _ {\text {s i n k}} \| K _ {\text {t o p}} \| K _ {\text {r c t}} \right], \tag {12}
$$

$$
\left[ V _ {\text {s i n k}} \| V _ {\text {c a n d}} \| V _ {\text {r c t}} \right]\rightarrow \left[ V _ {\text {s i n k}} \| V _ {\text {t o p}} \| V _ {\text {r c t}} \right].
$$

PC compresses only the intermediate tokens betweenthe sink and recent. This design preserves both the initialcontext $( K _ { \mathrm { s i n k } } , V _ { \mathrm { s i n k } } )$ and recent context $( K _ { \mathrm { r c t } } , V _ { \mathrm { r c t } } )$ withoutmodification, while compressing only the candidate tokens$( K _ { \mathrm { c a n d } } , V _ { \mathrm { c a n d } } )$ to retain the most important visual and con-textual information $( K _ { \mathrm { t o p } } , V _ { \mathrm { t o p } } )$ .

This compression occurs independently in each layer.Importantly, PC is applied only at the first diffusiontimestep $\mathit { t } = 1 0 0 0 $ ) when the cache reaches over its max-imum window length. The tokens selected at this ini-tial timestep remain fixed throughout subsequent denoisingsteps.

Figure 11 validates this design choice by demonstratingthat attention patterns remain consistent across timesteps.The tokens deemed important when generating frames 19–21 exhibit similar importance scores across different diffu-sion timesteps $( t \implies 1 0 0 0 , 7 5 0 , 5 0 0 , 2 5 0 )$ , confirming thatthe Top- $C$ selection at $t = 1 0 0 0$ captures tokens that re-main contextually relevant throughout the entire denoisingprocess.

Participative Compression Ablation. As shown in Fig-ure 3 in the main paper, Participative Compression (PC) canleverage both current denoising query tokens and clean pastquery tokens from previously generated frames to select theTop- $C$ candidates. We evaluate the effect of using each typeindependently versus combining them together.

Table 5 compares these three strategies. When Top- $C$is selected using only clean past tokens (Only Past), the

![](images/6195e107d600438e4f7c022437fa725b5b71c47a9885bf06383cc515528d84e5.jpg)



Figure 9. Qualitative results on different Attention Sink. The result shows that Deep Sink substantially outperforms both LongLive-style [31] and Rolling Forcing-style [17] attention sinks.



Table 5. Ablation on Participative Compression.


<table><tr><td>Method</td><td>Motion Smoothness</td><td>Overall Consistency</td><td>Image Quality</td></tr><tr><td>Only Denoising</td><td>97.86</td><td>20.44</td><td>68.24</td></tr><tr><td>Only Past</td><td>97.91</td><td>20.47</td><td>68.54</td></tr><tr><td>Both</td><td>98.27</td><td>20.54</td><td>69.31</td></tr></table>

method achieves an image quality of 68.54 and overall con-sistency of 20.47. When selection relies solely on cur-rently denoising tokens (Only Denoising), the noisy natureof these queries at the initial timestep $\mathit { t } = 1 0 0 0$ ) leads toslightly lower image quality (68.24) and motion smooth-ness (97.86), likely due to unstable token selection at theinitial denoising step. Combining both query types (Both)achieves the highest scores across all metrics, including mo-tion smoothness, overall consistency, and image quality.The clean past queries appear to provide relatively stableimportance estimates, while the current denoising querieshelp ensure the selected tokens remain relevant to the imme-diate generation context, suggesting complementary bene-fits from their combination.

# D. Denoising Query Is Not Just Random Noise

While the denoising queries at the initial timestep $\mathit { \Omega } ( t \ =$1000) are inherently noisy, they may still carry meaning-ful signal for identifying important tokens. Figure 11 sug-gests this by showing consistent attention patterns acrosstimesteps, but to more conclusively demonstrate the effec-tiveness of noisy queries, we directly compare Top- $C$ se-lection based on denoising queries versus Gaussian randomselection. For denoising query-based selection, we compute$Q K ^ { T }$ using only the currently denoising query tokens, then

select the Top- $C$ candidates. For Gaussian random selec-tion, we assign each candidate token a score sampled from$\mathcal { N } ( 0 , 1 )$ and select the Top- $C$ based on these random scores.

Figure 12 illustrates the stark difference. Random selec-tion exhibits severe scene repetition and context loss, as ran-domly chosen anchors fail to preserve coherent contextualinformation. In contrast, denoising query-based selectiongenerates context-aware videos with notably better subjectconsistency and context.

Figure 13 further validates this through token selec-tion frequency heatmaps over 1-minute generation, wherecolor intensity (white to dark purple) indicates selectionfrequency. The x-axis spans tokens 0–32,760: tokens 0–15,600 are Deep Sink, 15,600–28,080 are compression can-didates, and $^ { 2 8 , 0 8 0 + }$ are recent. Gaussian random selection(top) distributes uniformly across candidates, while denois-ing query-based selection (bottom) concentrates heavily onspecific positions, particularly immediately after the sinkboundary (15,600).

Notably, these high-frequency positions do not corre-spond to fixed frames, as tokens are evicted during com-pression, subsequent tokens shift into these slots. The con-centration at positions near 15,600 indicates that these po-sitional slots are consistently selected regardless of frameidentity, as they bridge established context (sink) and cur-rent generation (recent), serving as semantically importantanchors. This positional selectivity demonstrates meaning-ful contextual relationships rather than arbitrary noise.

We hypothesize this effectiveness stems from: (1) SelfForcing’s 4-step distilled diffusion enabling rapid conver-gence to meaningful attention patterns despite noisy queriesat $t = 1 0 0 0$ , and (2) per-layer KV caching allowing inde-pendent selection of semantically important tokens based

![](images/5f8d3e5ed6ff196fa3e714015e446af781ba86451abef5bcacc5aba457fb3c88.jpg)



Figure 10. Qualitative comparison of different sink sizes on 60-second videos. As the sink size decreases, degradation becomes moresevere. Once the sink size exceeds 10 frames, degradation is substantially reduced.


![](images/817f0417b120d35fe8fb67c2cd840b816dcc836d350f36390700b8af77ae082a.jpg)


![](images/db27b32ba3a038b8d12dd47f770533c2a5c8e0ff5686890e475730d04c8f130f.jpg)


![](images/891e7678d1acf9ed659c0f7639e9e70a3d1b3b6b307bd0ee898df13e18a1aa64.jpg)


![](images/4d7d7e38836f1414a86883136b1e7ab003c77064e72e43e8fc43e9bacaf518f7.jpg)



Figure 11. Attention weight consistency across diffusion timesteps. Query-averaged attention weights showing how each key frame isattended when generating the last chunk (frames 19–21) at different denoising timesteps. The consistent attention patterns across timesteps(1000, 750, 500, 250, and the final clean KV cache) demonstrate that Top- $C$ tokens selected at the initial timestep $t = 1 0 0 0$ ) remain validand contextually relevant throughout the entire denoising process.


![](images/f732f6fa4db08439384d3b2405e012943922418d59b234b43eeba3a346e0808f.jpg)



Figure 12. Qualitative comparison: Random Top- $C$ vs. Denoising Query Top-C. Gaussian random selection causes severe artifactsduring compression - faces abruptly rotate, heads appear floating in mid-air, and random context drift occurs, resulting in incoherentscene transitions. In contrast, denoising query-based selection maintains subject consistency with natural emergent camera movements andpreserves contextual coherence throughout the generation.


on layer-specific contextual relevance.

# E. FPS measurements

We evaluate inference throughput on a single NVIDIAH100 GPU when generating 60-second videos. Table 6demonstrates that Deep Forcing maintains throughput com-

parable to baseline Self Forcing, achieving 15.75 FPS ver-sus 15.78 FPS. Despite the computational overhead of com-pression, our method balances two competing factors: (1)compressing from 21 to 16 frames requires additional com-putation, but (2) generating subsequent frames with only 16cached frames incurs lower attention costs compared to full

![](images/21616c6df84371d23befbc93502e183fa68ea7a805f6ca8efda3538488c10f7d.jpg)



Figure 13. Token-wise Top- $C$ selection frequency heatmap during 1-minute generation. Color intensity ranges from white (rarelyselected) to dark purple (frequently selected as Top- $C$ ), indicating how often each token is reused throughout the generation. The x-axisspans tokens 0–32,760, where 0–15,600 are Deep Sink tokens, 15,600–28,080 are candidates for compression, and $^ { 2 8 , 0 8 0 + }$ are recenttokens. Gaussian random selection (top) distributes selections uniformly across candidate tokens, whereas denoising query-based selection(bottom) concentrates heavily on specific semantically important tokens—particularly those immediately after the sink boundary—thateffectively bridge established and newly formed context.



Table 6. Throughput Comparison on a single H100 GPU. La-tency is measured after first rolling.


<table><tr><td>Method</td><td>FPS</td><td>Latency(Min/Max)</td></tr><tr><td>Self Forcing [12]</td><td>15.78</td><td>0.770 / 0.776s</td></tr><tr><td>Deep Forcing (Ours)</td><td>15.75</td><td>0.747 / 0.797s</td></tr></table>

attention over 21 frames.

The latency range after first rolling in Table 6 reflectsthis trade-off. While Deep Forcing exhibits a wider la-tency range (0.747s to 0.797s) compared to the baseline(0.770s to 0.776s), the average latencies are nearly identi-cal, demonstrating that compression overhead is effectivelybalanced by reduced attention costs. In practice, through-put oscillates between compression phases (slightly slower)and generation phases (slightly faster) as the cache alter-nates between 21 and 16 frames. These fluctuations averageto nearly identical performance as the baseline, demonstrat-ing that our compression mechanism effectively amortizesits overhead, enabling long-horizon generation with mini-mal performance penalty.

# F. More Qualitative Results

Additional qualitative results of our method are presentedin Figure 14, and Figure 15. These examples clearly showthat our training-free Deep Sink and Participative Com-pression framework produces results on par with training-based methods.

# G. User Study Protocol

To perform human evaluation, we conducted a user studybased on the Two-Alternative Forced Choice (2AFC) pro-tocol [2, 3]. For each question, participants were pre-sented with two videos generated from the same promptand instructed to choose which video they preferred ac-

cording to four evaluation criteria: (1) Color Consistency -Which video maintains more consistent color and exposurethroughout, without sudden shifts in brightness, saturation,or color tone? (2) Dynamic Motion - Which video exhibitsmore dynamic and varied motion, including both subjectmovement and camera movement? (3) Subject Consistency- Which video maintains better visual consistency of themain subject throughout its duration? (Consider comparingthe beginning and end of each video.) (4) Overall Quality -Overall, which video appears more realistic, natural, and ofhigher quality?

Each participant evaluated 16 video pairs compar-ing Deep Forcing against each of the four baselines(CausVid [35], Self Forcing [12], LongLive [31], andRolling Forcing [17]). For each baseline, participants wereshown 4 pairwise comparisons using different prompts,with all 16 prompts being non-overlapping within each par-ticipant’s session. These prompts were randomly sampledfrom a pool of 20 total prompts. With 24 total participants,this yielded 384 total video comparisons (24 participants $\times$16 pairs), the results of which are shown in Table 2. Thepresentation order of videos was randomized, and partici-pants were not informed which model generated each video.This design ensured balanced evaluation across all baselinemodels while avoiding prompt repetition within individualsessions. The user interface is shown in Figure 17.

# H. Additional Attention Visualization

We provide additional attention head visualizations (Fig.16)beyond those shown in Fig.4 from Section 4.2. This deepattention pattern with substantial weight on both initial andintermediate tokens, emerges consistently and pervasivelyacross layers and heads, rather than being only one or twospecific heads, supporting the hypothesis that deep sinks arefundamental to Self Forcing [12].

![](images/180450b61b1ae17d708f20cf38026891a552a6f4cc0ba2bab34ac32612ed3f2b.jpg)



Figure 14. Qualitative results on 30-second videos. Frame-by-frame comparison across different methods for two representative prompts.Deep Forcing (training-free) achieves temporal consistency and visual quality comparable to training-based baselines (CausVid [35], SelfForcing [12], LongLive [31], Rolling Forcing [17]) while generating more dynamic content with greater subject consistency.


![](images/cd46549a21771f10eba659efcaa1a4eb0dfee47691f2351263706b6de13afd9f.jpg)



Figure 15. Qualitative results on 60-second videos. Frame-by-frame comparison across different methods for two representative prompts.Deep Forcing (training-free) achieves temporal consistency and visual quality comparable to training-based baselines (CausVid [35], SelfForcing [12], LongLive [31], Rolling Forcing [17]) while generating more dynamic content with greater subject consistency.


![](images/3a28edc1126521568e6788fd17bdbfd367ce6c76a36bed4c31195f6124e1270e.jpg)


![](images/2197da012637dcc614b193afea17c1c38f9f6a990209649d3624f3b59118f2b4.jpg)


![](images/05897ade4e575d9188f7d744125fc106ff8be5442f0396d7d361862681c28f98.jpg)


![](images/240cad7e2762562a8f50d63946833aa42acb42d5f1e77aad281f0f3b2797de7f.jpg)


![](images/e07c4d4cbec2d0d0789ed6e3ef5dfb965feb30af5c3f3038ba3b7456c708f7ab.jpg)



Key Frames


![](images/fddaa7d90928199f9cf351040ec13c6eda62c2d125b6da8dc678f1dc5bd9dd9d.jpg)


![](images/37524a20c22d5421b9d34dbed140df066e7b7f7d1224e245e8ef307c9f124c74.jpg)


![](images/8d6ccd4bfead18b3cf49697a8f8a168706273ec4925fd1e6b5b4fa3937a4f447.jpg)


![](images/8465051d9cdc77b394fcbf15ace1913117ce0cb8a5b630052139426914a4102f.jpg)


![](images/67f76c6ae8462b641300b0298257b566e1ad1f3f8accbab27191223b7de37ae3.jpg)



Key Frames



Figure 16. Attention weight distribution across earlier frames. Query-averaged attention showing how the last chunk (frames 19-21)attends to earlier KV cache entries (frames 0-18). Each frame consists of 1,560 tokens (spatially arranged latent patches). We visualizemultiple attention heads from different layers, demonstrating that substantial attention to intermediate tokens is consistent across layersand heads.


# Text Prompt:

Awintersceneinowyforestherealiterofplayfuldeetrieverpuppiesemergefromtheow.heieadspopoutterfuffyfurglistenituhtndatailsulyeovedoitoetdingantpsnow.OnepupyiurinitsoseineoilaotehasesallballthatolledarbyTackgrondowseevergreentredtoeaneainiispdlditioakefllnntlyofromaslightlyelevatedangle,capturingthelivelyandenergeticmoment.

![](images/0dad1e802978e5603b8ded79f41ec38a7424dd7944a9fe8eb30849f363d9e207.jpg)


Please answerallquestions below after watching thevideo completely. Watch thevideomultiple timesif needed to evaluatedifferent aspects.

1.[Color Consistency] Which video maintains more consistent color and exposure throughout,without suddenshifts in brightness, saturation,or color tone?

O Model A

O Model B

2.[Dynamic Motion] Which video exhibits more dynamic and varied motion,including both subject movement andcamera movement?

O Model A

O Model B

3.[Subject Consistency] Which video maintains better visual consistency ofthe main subject throughout itsduration?(Consider comparing the beginning and end of each video)

OModelA

4.[Overall Quality] Overall, which video appears more realistic, natural,andof higheroverallquality?

OModelA

O Model B

NEXT VIDEO →


Figure 17. Example of the user interface for the user study.
