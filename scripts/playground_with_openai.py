import openai


# Set the prompt you want the model to complete
prompt = "Once upon a time, in a land far away,"

# Call the OpenAI API to generate a completion
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=50,
    n=1,
    stop=None,
    temperature=0.5,
)

# Extract and print the generated text
generated_text = response.choices[0].text.strip()
print(generated_text)
