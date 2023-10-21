library(tidyverse)
library(brms)
library(tidyboot)
library(tidyjson)
library(tidybayes)
library(patchwork)
library(GGally)
library(cowplot)
library(BayesFactor)
library(aida)   # custom helpers: https://github.com/michael-franke/aida-package
library(faintr) # custom helpers: https://michael-franke.github.io/faintr/index.html
library(cspplot)

##################################################

# these options help Stan run faster
options(mc.cores = parallel::detectCores(),
        brms.backend = "cmdstanr")

# use the CSP-theme for plotting
theme_set(theme_csp())

# global color scheme from CSP
project_colors = cspplot::list_colors() |> pull(hex)

# setting theme colors globally
scale_colour_discrete <- function(...) {
  scale_colour_manual(..., values = project_colors)
}
scale_fill_discrete <- function(...) {
  scale_fill_manual(..., values = project_colors)
}

##################################################

# # explore variants of label_score
e <- read_csv("processed_data/label_scores_argmax_wide.csv")
# e <- read_csv("processed_data/string_scores_argmax_wide.csv")

# all of these are expected to be true for labels (so we expect the mean to be 1)
# for sentences we expect the outcomes below
mean(e$sentence_cond_probs == e$mean_sentence_cond_probs)       # FALSE
mean(e$sentence_surprisal == e$mean_sentence_surprisal)         # FALSE
mean(e$sentence_mi == e$sentence_mi_surprisal)                  # TRUE
mean(e$sentence_cond_probs == e$sentence_surprisal)             # TRUE

# this is expected to be false for sentences (if labels are single tokens, we expect TRUE as well)
mean(e$mean_sentence_cond_probs == e$mean_sentence_surprisal)

e[which(e$sentence_cond_probs != e$mean_sentence_cond_probs),]
e[which(e$sentence_surprisal != e$mean_sentence_surprisal),]
e[which(e$sentence_cond_probs != e$sentence_surprisal),]     
e[which(e$mean_sentence_cond_probs != e$mean_sentence_surprisal),]

##############################################
## new data
##############################################

d_human <- 
  rbind(
    read_csv("../human_data/Human_CoherenceInference.csv") |> mutate(phenomenon = "coherence"),
    read_csv("../human_data/Human_Deceits.csv") |> mutate(phenomenon = "deceits"),
    read_csv("../human_data/Human_Humour.csv") |> mutate(phenomenon = "humour"),
    read_csv("../human_data/Human_IndirectSpeech.csv") |> mutate(phenomenon = "indirect_speech"),
    read_csv("../human_data/Human_Irony.csv") |> mutate(phenomenon = "irony"),
    read_csv("../human_data/Human_Maxims.csv") |> mutate(phenomenon = "maxims"),
    read_csv("../human_data/Human_Metaphor.csv") |> mutate(phenomenon = "metaphor")
  ) |> 
  group_by(phenomenon) |> 
  summarize(
    k_human = sum(Correct),
    N_human = n(),
    accuracy_human = mean(Correct)
  ) |> 
  rename(condition = phenomenon)

epsilon <- 0.000000001

d_final <- read_csv("llmlink_long_final.csv")

d_final_prepped <- d_final  |> 
  group_by(model, method, score, condition) |> 
  summarize(
    k_LLM = sum(target),
    N_LLM = n(),
    accuracy_LLM = mean(target)
  ) |> 
  ungroup() |> 
  full_join(d_human, by = "condition") |> 
  mutate(
    accuracy_LLM_corrected = ifelse(accuracy_LLM == 0, epsilon, ifelse(accuracy_LLM == 1, 1-epsilon, accuracy_LLM))) |> 
  group_by(model, method, score, condition) |> 
  mutate(
    LLH = dbinom(x = k_human, size = N_human, prob = accuracy_LLM_corrected, log = T)
    # MSE = mean((accuracy_LLM - k_human/N_human)^2)
  ) |> 
  ungroup() |> 
  group_by(model, method, score) |> 
  summarize(
    LLH = sum(LLH),
    accuracy = sum(k_LLM) / sum(N_LLM)
    # MSE = mean((accuracy_LLM - k_human/N_human)^2)
  ) |> 
  ungroup() |> 
  mutate(
    model = case_when(
      model == "google/flan-t5-xl" ~ "FLAN-T5",
      model == "gpt-3.5-turbo-instruct" ~ "GPT-instruct",
      model == "meta-llama/Llama-2-7b-hf" ~ "LLaMA",
      model == "text-davinci-002" ~ "GPT-davinci",
      model == "text-embedding-ada-002" ~ "GPT-davinci",
    )
  ) |> 
  mutate(
    score = case_when(
      score == "sentence_cond_probs"  ~  "cond. prob.",
      score == "mean_sentence_cond_probs"  ~  "avg. cond. prob.",
      score == "mean_sentence_surprisal"  ~  "avg. surpisal",
      score == "sentence_mi"  ~  "emp. MI",
      TRUE ~ score
    )
  ) |> 
  arrange(method, score, model) |> 
  select(method, score, model, LLH, accuracy)


variant_order <- c(
  "free - free",
  "string_score - cond. prob.",
  "string_score - avg. cond. prob.",
  "string_score - avg. surpisal",
  "string_score - emp. MI",
  "string_score - surprisal_decrease_ratio",
  "label_score - cond. prob.",
  "label_score - emp. MI",
  "label_score - surprisal_decrease_ratio",
  "rating - rating",
  "embedding_similarity - cosine"
)

d_final_prepped <- d_final_prepped |> 
  mutate(
    variant = str_c(method, " - ", score),
    variant = factor(variant, levels = variant_order),
    method  = factor(method, levels = c("free", "string_score", "label_score", "rating", "embedding_similarity")),
    method  = fct_recode(method, "string" = "string_score", "label" = "label_score", "embedding" = "embedding_similarity"),
    # LLH = ifelse(LLH > -100000, LLH, 0),
    LLH = ifelse(LLH > -100000, LLH + 68768.470 + 500, 500) 
  ) |> 
  filter(! is.na(variant)) |>
  rename(
    `acc.rcy` = accuracy,
    `LLH diff.` = LLH
    )

d_final_prepped |> 
  pivot_longer(cols = c(`LLH diff.`, `acc.rcy`)) |> 
  ggplot(aes(x = variant, y = value, fill = method)) +
  geom_col() +
  xlab("") + ylab("") +
  facet_grid(name ~ model, scales = "free") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  scale_x_discrete(labels = c("free", "str::optProb", "str::avgOptProb", "str::avgNegSurp", "str::priorCorrOptProb", "str::SurpRedFct", 
                              "lbl::optProb", "str::priorCorrOptProb", "str::SurpRedFct", "rating", "embedding"))

ggsave("results-combined.pdf", width = 12, height = 4.75, scale = 1)












