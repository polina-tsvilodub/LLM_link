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

## read and wrangle  data

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
  )

epsilon <- 0.000000001

aar <- read_csv("processed_data/all_accuracy_raw.csv") |> 
  filter(metric %in% c( "free", "rating")) |> 
  mutate(metric_formula = "sentence_cond_probs")
  
d_labels <- read_csv("processed_data/label_scores_argmax_long.csv")  |> mutate(seed = 0)
d_string <- read_csv("processed_data/string_scores_argmax_long.csv") |> mutate(seed = 0)

d_raw <- rbind(aar, d_labels, d_string)

d <- 
  d_raw  |> 
  filter(! metric_formula %in% c("sentence_mi_surprisal", "sentence_surprisal") ) |> 
  group_by(model_name, phenomenon, metric, metric_formula) |> 
  summarize(
    k_LLM = sum(dependent_variable),
    N_LLM = n(),
    accuracy_LLM = mean(dependent_variable)
    ) |> 
  ungroup() |> 
  full_join(d_human, by = "phenomenon") |> 
  mutate(
    accuracy_LLM_corrected = ifelse(accuracy_LLM == 0, epsilon, ifelse(accuracy_LLM == 1, 1-epsilon, accuracy_LLM))) |> 
  group_by(model_name, metric) |> 
  mutate(
    LLH = dbinom(x = k_human, size = N_human, prob = accuracy_LLM_corrected, log = T),
    MSE = mean((accuracy_LLM - k_human/N_human)^2)
  )
  

###################################################

## testing 

# View(d)

results <- d |> 
  group_by(model_name, metric, metric_formula) |> 
  summarize(
    LLH = sum(LLH),
    accuracy = sum(k_LLM) / sum(N_LLM),
    MSE = mean((accuracy_LLM - k_human/N_human)^2)
  ) |> 
  ungroup() |> 
  mutate(
    model = case_when(
      model_name == "google/flan-t5-xl" ~ "FLAN-T5",
      model_name == "gpt-3.5-turbo-instruct" ~ "GPT-instruct",
      model_name == "meta-llama/Llama-2-7b-hf" ~ "LLaMA",
      model_name == "text-davinci-002" ~ "GPT-davinci",
    )
  ) |> 
  mutate(
    metric_formula = case_when(
      metric_formula == "sentence_cond_probs"  ~  "cond. prob.",
      metric_formula == "mean_sentence_cond_probs"  ~  "avg. cond. prob.",
      metric_formula == "mean_sentence_surprisal"  ~  "avg. surpisal",
      metric_formula == "sentence_mi"  ~  "emp. MI"
    )
  ) |> 
  arrange(metric, metric_formula, model) |> 
  select(metric, metric_formula, model, LLH, accuracy, MSE)

# View(results)

# LaTeX tables
# results |> 
#   filter(metric %in% c("free", "rating")) |> 
#   select(-metric_formula) |> 
#   knitr::kable(booktabs = T,format = "latex")
# results |> 
#   filter(! metric %in% c("free", "rating")) |> 
#   knitr::kable(booktabs = T,format = "latex")


# results plot

variant_order <- c(
  "free - cond. prob.",
  "string_score - cond. prob.",
  "string_score - avg. cond. prob.",
  "string_score - avg. surpisal",
  "string_score - emp. MI",
  "label_score - cond. prob.",
  "label_score - avg. cond. prob.",
  "label_score - avg. surpisal",
  "label_score - emp. MI",
  "rating - cond. prob."
)

results |> 
  mutate(
    variant = str_c(metric, " - ", metric_formula),
    variant = factor(variant, levels = variant_order),
    metric = factor(metric, levels = c("free", "string_score", "label_score", "rating")),
    metric = fct_recode(metric, "string" = "string_score", "label" = "label_score"),
    # LLH = ifelse(LLH > -100000, LLH, 0),
    LLH = ifelse(LLH > -100000, LLH + 68768.470 + 500, 500) 
  ) |> 
  rename(`acc.rcy` = accuracy) |> 
  pivot_longer(cols = c(LLH, `acc.rcy`)) |> 
  ggplot(aes(x = variant, y = value, fill = metric)) +
  geom_col() +
  xlab("") + ylab("") +
  facet_grid(name ~ model, scales = "free") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  scale_x_discrete(labels = c("free", "str::condProb", "str::avgCondProb", "str::avgSurprial", "str::empMI", 
                              "lbl::condProb", "lbl::avgCondProb", "lbl::avgSurprial", "lbl::empMI", "rating"))

ggsave("results-combined.pdf", width = 12, height = 4.2, scale = 1)

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




