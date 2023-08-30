from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

input_text = "Gib die am meisten plausible Fortsetzung nach dem folgenden Kontext: Peter ist sehr gut in Mathe. Heute hat seine Klasse eine Mathe Prüfung geschrieben und die Ergebnisse sind da. Ausgerechnet Peter ..."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_new_tokens = 100, temperature=1.0)
print(tokenizer.decode(outputs[0]))
print(outputs)