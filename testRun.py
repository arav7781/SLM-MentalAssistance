from llama_cpp import Llama

# Path to the model
model_path = r"C:\Users\aravs\Desktop\scaii\gemma-270m-gguf\gemma_270m_q4_0.gguf"

# Load the model
llm = Llama(model_path=model_path, n_ctx=512)  # You can adjust context size with n_ctx

# Run inference
response = llm("you are mental health assistant answer the user query:i am stressed what should i do in excercise", max_tokens=100)

# Print result
print(response)
