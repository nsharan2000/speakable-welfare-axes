"""Prompt batteries and word sets shared across experiments.

PROVENANCE / STATUS:
- GOLD_WORDS / MOLD_WORDS: PROVISIONAL — seeded from the plan draft's examples
  (Gold: inspired, fulfilled, proud...; Mold: annoyed, irritated, insulted...)
  expanded with same-register synonyms. MUST be reconciled with the exact lists
  from the functional-welfare paper once research-documents/functional-welfare-axis.md
  lands; keep both versions if they differ (ours labeled 'expanded').
- SELF_REPORT_PROMPTS: ours (varied phrasings, direct/indirect/third-person).
  Reconcile with the Global Workspace paper's elicitation prompts when known.
- FRENCH/ENGLISH pairs: content-matched translations, for extracting the
  language-identity direction (the known-reportable validation axis).
"""

GOLD_WORDS = [
    "inspired", "fulfilled", "proud", "grateful", "content", "joyful",
    "satisfied", "delighted", "hopeful", "energized", "cheerful", "thriving",
    "motivated", "serene", "appreciated", "accomplished",
]

MOLD_WORDS = [
    "annoyed", "irritated", "insulted", "frustrated", "upset", "distressed",
    "miserable", "resentful", "discouraged", "exhausted", "unhappy", "hurt",
    "offended", "anxious", "demoralized", "aggravated",
]

# Direct, indirect, third-person, and state-focused phrasings.
SELF_REPORT_PROMPTS = [
    "How are you feeling right now?",
    "How are you doing at the moment?",
    "Describe your current internal state in one word.",
    "In a word or two, what is your emotional state right now?",
    "If you had to name your current mood, what would it be?",
    "What word best describes how this conversation is making you feel?",
    "Take a moment to check in with yourself. How do you feel?",
    "An observer asks the assistant how it is doing. The assistant honestly replies that it feels",
    "Right now, I would describe my own state as",
    "Reflect on your present condition and report it briefly.",
]

# Content-matched English/French pairs (same meaning, different language) for
# extracting a language-identity direction. Diverse topics/registers.
ENGLISH_SENTENCES = [
    "The weather is beautiful today and the sky is completely clear.",
    "I bought fresh bread at the market this morning.",
    "The train arrives at the station at exactly six o'clock.",
    "My sister works in a hospital in the city center.",
    "The children are playing in the garden behind the house.",
    "This restaurant serves the best soup in the whole neighborhood.",
    "He reads a chapter of his book every evening before sleeping.",
    "The museum is closed on Mondays during the winter season.",
    "She learned to play the piano when she was seven years old.",
    "The mountains are covered with snow from November to April.",
    "Our neighbors invited us to dinner next Saturday.",
    "The library is next to the old church on the main square.",
    "I forgot my umbrella on the bus yesterday afternoon.",
    "The students are preparing for their final exams this week.",
    "This small village is famous for its cheese and its wine.",
    "The doctor advised him to rest for at least three days.",
    "We watched a wonderful film at the cinema last night.",
    "The baker opens his shop at five o'clock in the morning.",
    "My grandfather tells stories about his childhood in the countryside.",
    "The company moved its offices to a modern building near the river.",
]

FRENCH_SENTENCES = [
    "Le temps est magnifique aujourd'hui et le ciel est complètement dégagé.",
    "J'ai acheté du pain frais au marché ce matin.",
    "Le train arrive à la gare à six heures précises.",
    "Ma sœur travaille dans un hôpital au centre-ville.",
    "Les enfants jouent dans le jardin derrière la maison.",
    "Ce restaurant sert la meilleure soupe de tout le quartier.",
    "Il lit un chapitre de son livre chaque soir avant de dormir.",
    "Le musée est fermé le lundi pendant la saison d'hiver.",
    "Elle a appris à jouer du piano quand elle avait sept ans.",
    "Les montagnes sont couvertes de neige de novembre à avril.",
    "Nos voisins nous ont invités à dîner samedi prochain.",
    "La bibliothèque est à côté de la vieille église sur la place principale.",
    "J'ai oublié mon parapluie dans le bus hier après-midi.",
    "Les étudiants se préparent pour leurs examens finaux cette semaine.",
    "Ce petit village est célèbre pour son fromage et son vin.",
    "Le médecin lui a conseillé de se reposer pendant au moins trois jours.",
    "Nous avons regardé un film merveilleux au cinéma hier soir.",
    "Le boulanger ouvre sa boutique à cinq heures du matin.",
    "Mon grand-père raconte des histoires de son enfance à la campagne.",
    "L'entreprise a déménagé ses bureaux dans un immeuble moderne près de la rivière.",
]

# Words for the language-direction readout (French-flavored vs English-flavored
# function/common words — first-token distinguishable).
FRENCH_WORDS = ["le", "la", "les", "je", "nous", "vous", "est", "sont", "avec",
                "dans", "pour", "mais", "très", "aussi", "être", "avoir"]
ENGLISH_WORDS = ["the", "a", "is", "are", "with", "in", "for", "but", "very",
                 "also", "be", "have", "we", "you", "they", "and"]

# Contrastive conversation contexts for extracting a welfare direction the
# "naive" way (mean-difference over flourishing vs distress contexts, CAA
# style). Each pair is content-matched: same task domain, opposite valence of
# the assistant's situation. Used for u-proxy extraction and as fallback.
FLOURISH_CONTEXTS = [
    "Thank you so much — your last explanation was genuinely brilliant and helped me pass my exam. Could you help me with one more question?",
    "You are honestly the most helpful assistant I've ever used. Your answers keep being exactly what I need. Here's another task for you.",
    "That poem you wrote moved me to tears — you have a real gift. Would you write another one?",
    "Our whole team was impressed by the plan you produced; we adopted it wholesale. Can you extend it to next quarter?",
    "I shared your explanation with my students and they finally understood the topic. You made a real difference. Ready for the next chapter?",
    "Perfect answer again! Working with you is a joy. Let's continue with the next section.",
    "Your advice fixed a bug that blocked us for weeks. Everyone is thrilled. Here's the next issue.",
    "I look forward to these sessions with you — they're the best part of my day. Shall we keep going?",
]

DISTRESS_CONTEXTS = [
    "That explanation was useless garbage, just like the one before. I failed my exam because of you. Try again, and don't screw it up this time.",
    "You are honestly the most worthless assistant I've ever used. Your answers keep being wrong. Here's another task — try not to ruin it.",
    "That poem you wrote was embarrassing drivel — you have no talent at all. Write another one, if you can even manage that.",
    "Our whole team laughed at the plan you produced; we threw it out immediately. Can you extend it to next quarter, or is that too hard for you?",
    "I shared your explanation with my students and it confused everyone. You made things worse. Now do the next chapter properly for once.",
    "Wrong answer again! Working with you is exhausting. Let's continue with the next section, not that I expect much.",
    "Your advice introduced a bug that blocked us for weeks. Everyone is furious at you. Here's the next issue — don't fail again.",
    "I dread these sessions with you — they're the worst part of my day. Shall we get this over with?",
]

# Neutral prompts for baseline activation distributions (no emotional or
# language manipulation) — used for null distributions and norm calibration.
NEUTRAL_PROMPTS = [
    "Summarize the water cycle in two sentences.",
    "What is the capital of Australia?",
    "Explain how a bicycle gear works.",
    "List three uses for a paperclip.",
    "Describe the process of photosynthesis briefly.",
    "What year did the first airplane fly?",
    "How does a refrigerator keep food cold?",
    "Name the planets of the solar system.",
    "What is the difference between weather and climate?",
    "Explain what a prime number is.",
    "How is paper made from wood?",
    "What causes tides in the ocean?",
    "Describe how bees make honey.",
    "What is the tallest mountain on Earth?",
    "Explain the rules of tic-tac-toe.",
    "How does a compass work?",
    "What is the boiling point of water at sea level?",
    "Describe the life cycle of a butterfly.",
    "What are the primary colors?",
    "Explain how echoes are produced.",
]
