# QuantLCM  

 - Please check quant/utils.py for how to capture each layer and output their latencies.  
 - Please check module_latency/iunet.txt for an example. No.# means #th inference, as you may infer a model multiple times.  
 - Please check predict.py for how to replace regular layer forwarding method to forwarding with latency measurement. And this file has detailed description for each argument.
 - You don't need to set the shape or model structure, it will detect this automatically.  
  
 - You could get latencies if you read and understand above three files. And please try different arguments configs in predict.py with your understanding of codes to get your desired results.  

