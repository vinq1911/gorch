//go:build darwin

package main

import "fmt"

// Deterministic TEXT-replay seed prompts (plan 0008 §4.3): short
// factual/conversational questions whose greedy base-model answers
// become the frozen-brain replay targets. The lists below expand
// combinatorially in a fixed order; buildPrompts(n) returns the first
// n (the default 500 draws on ~560 generated prompts).

var promptCountries = []string{
	"France", "Japan", "Brazil", "Canada", "Egypt", "Kenya", "Norway", "Spain",
	"Thailand", "Mexico", "India", "Italy", "Germany", "Australia", "Peru",
	"Finland", "Greece", "Turkey", "Poland", "Chile", "Portugal", "Austria",
	"Ireland", "Sweden", "Denmark", "Argentina", "Morocco", "Vietnam",
	"South Korea", "Indonesia", "Netherlands", "Switzerland", "Hungary",
	"Czechia", "Croatia", "Iceland", "New Zealand", "Colombia", "Cuba", "Ghana",
}

var promptMathPairs = [][2]int{
	{2, 3}, {4, 5}, {6, 7}, {8, 9}, {3, 7}, {5, 8}, {9, 4}, {7, 6}, {12, 11},
	{15, 4}, {13, 8}, {17, 5}, {21, 9}, {14, 7}, {18, 3}, {25, 6}, {32, 9},
	{11, 12}, {19, 7}, {23, 8}, {16, 5}, {27, 4}, {31, 2}, {35, 6}, {41, 3},
	{45, 8}, {52, 7}, {29, 9}, {33, 5}, {38, 4},
}

var promptCategories = []string{
	"fruits", "animals", "colors", "countries in Europe", "musical instruments",
	"planets", "vegetables", "sports", "programming languages", "rivers",
	"mountains", "birds", "fish", "trees", "flowers", "metals", "gemstones",
	"dances", "board games", "professions", "weather phenomena", "sea creatures",
	"breakfast foods", "farm animals", "kitchen utensils", "items of clothing",
	"parts of the body", "school subjects", "modes of transport", "shapes",
}

var promptConcepts = []string{
	"gravity", "photosynthesis", "democracy", "inflation", "evaporation",
	"friction", "electricity", "magnetism", "an ecosystem", "a vaccine",
	"a metaphor", "an eclipse", "a volcano", "a glacier", "the water cycle",
	"a computer", "the internet", "a recipe", "a calendar", "a compass",
	"a thermometer", "an island", "a peninsula", "a desert", "a rainforest",
	"a library", "a museum", "an orchestra", "a marathon", "a telescope",
	"a microscope", "a satellite", "a bridge", "a lighthouse", "a windmill",
	"a habit", "a proverb", "a rhyme", "an echo", "a shadow",
}

var promptAntonyms = []string{
	"hot", "fast", "big", "early", "happy", "light", "loud", "hard", "old",
	"clean", "empty", "wide", "brave", "cheap", "smooth", "sharp", "deep",
	"strong", "wet", "bright", "tall", "thick", "rare", "calm", "generous",
}

var promptColors = []string{
	"a ripe banana", "the sky on a clear day", "fresh grass", "a ripe tomato",
	"snow", "coal", "a carrot", "a lemon", "an eggplant", "chocolate",
	"a flamingo", "the sun at noon", "a blueberry", "rust", "an emerald",
	"a polar bear", "a school bus", "a stop sign", "milk", "a pumpkin",
}

var promptComparisons = [][2]string{
	{"an elephant", "a mouse"}, {"the sun", "the moon"}, {"a whale", "a dolphin"},
	{"a mountain", "a hill"}, {"an ocean", "a lake"}, {"a jet", "a bicycle"},
	{"a tree", "a flower"}, {"a country", "a city"}, {"a novel", "a poem"},
	{"a marathon", "a sprint"}, {"a castle", "a cottage"}, {"a river", "a stream"},
	{"a planet", "an asteroid"}, {"a ship", "a canoe"}, {"a bear", "a fox"},
	{"a truck", "a scooter"}, {"a cathedral", "a chapel"}, {"an oak", "a sapling"},
	{"a lion", "a housecat"}, {"a glacier", "an icicle"},
}

var promptYesNo = []string{
	"Is the sun a star?", "Is a tomato a fruit?", "Can penguins fly?",
	"Is water made of hydrogen and oxygen?", "Do spiders have six legs?",
	"Is the Pacific the largest ocean?", "Are dolphins fish?",
	"Is zero an even number?", "Do plants need sunlight to grow?",
	"Is Mount Everest the tallest mountain on Earth?", "Can sound travel in space?",
	"Is a whale a mammal?", "Do bees make honey?", "Is ice less dense than water?",
	"Are bats blind?", "Is the Great Wall of China visible from space?",
	"Do all birds migrate?", "Is lightning hotter than the sun's surface?",
	"Can ostriches fly?", "Is chess older than checkers?",
}

var promptDays = []string{
	"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}

var promptAdvice = []string{
	"stay focused while studying", "start learning to cook", "sleep better",
	"keep a plant alive", "remember people's names", "save money each month",
	"make a good first impression", "learn a new language", "stay calm before a talk",
	"organize a small workspace",
}

var promptMonths = []string{
	"January", "February", "March", "April", "May", "June", "July",
	"August", "September", "October", "November", "December",
}

var promptAnimalLegs = []string{
	"a spider", "a dog", "a chicken", "an ant", "a horse", "a crab", "a snake",
	"a butterfly", "a cow", "an octopus", "a kangaroo", "a bee", "a cat",
	"a flamingo", "a beetle", "a sheep", "a duck", "a lizard", "a goat", "a pig",
}

var promptNouns = []string{
	"child", "mouse", "leaf", "box", "city", "wolf", "tooth", "foot", "berry",
	"knife", "hero", "potato", "goose", "man", "woman", "sheep", "deer",
	"library", "party", "wish", "brush", "fox", "bench", "story", "key",
}

var promptMisc = []string{
	"Count from one to five.",
	"Say the vowels of the English alphabet.",
	"What is the largest planet in our solar system?",
	"What is the smallest prime number?",
	"How many minutes are in an hour?",
	"How many hours are in a day?",
	"What language is spoken in Brazil?",
	"What do bees collect from flowers?",
	"What instrument has 88 keys?",
	"How many sides does a hexagon have?",
	"What gas do plants absorb from the air?",
	"What is frozen water called?",
	"How many continents are there?",
	"What is the chemical symbol for gold?",
	"How many strings does a violin have?",
}

// buildPrompts returns the first n prompts of the fixed expansion.
func buildPrompts(n int) []string {
	var out []string
	add := func(s string) { out = append(out, s) }

	for _, c := range promptCountries {
		add(fmt.Sprintf("What is the capital of %s?", c))
	}
	for _, p := range promptMathPairs {
		add(fmt.Sprintf("What is %d plus %d?", p[0], p[1]))
	}
	for _, p := range promptMathPairs {
		add(fmt.Sprintf("What is %d times %d?", p[0], p[1]))
	}
	for _, cat := range promptCategories {
		add(fmt.Sprintf("Name three %s.", cat))
	}
	for _, con := range promptConcepts {
		add(fmt.Sprintf("Explain in one sentence what %s is.", con))
	}
	for _, w := range promptAntonyms {
		add(fmt.Sprintf("Give one word that means the opposite of %s.", w))
	}
	for _, x := range promptColors {
		add(fmt.Sprintf("What color is %s?", x))
	}
	for _, p := range promptComparisons {
		add(fmt.Sprintf("Which is larger, %s or %s?", p[0], p[1]))
	}
	for _, q := range promptYesNo {
		add(q + " Answer briefly.")
	}
	for _, d := range promptDays {
		add(fmt.Sprintf("What day comes after %s?", d))
	}
	for _, a := range promptAdvice {
		add(fmt.Sprintf("Give one short tip to %s.", a))
	}
	for _, c := range promptCountries {
		add(fmt.Sprintf("Name one famous landmark in %s.", c))
	}
	for _, cat := range promptCategories {
		add(fmt.Sprintf("What is your favorite of all the %s? Answer in one sentence.", cat))
	}
	for _, w := range promptAntonyms {
		add(fmt.Sprintf("Use the word %q in a short sentence.", w))
	}
	for _, d := range promptDays {
		add(fmt.Sprintf("Is %s a weekend day? Answer briefly.", d))
	}
	for _, m := range promptMonths {
		add(fmt.Sprintf("What month comes after %s?", m))
	}
	for _, m := range promptMonths {
		add(fmt.Sprintf("How many days are in %s?", m))
	}
	for _, a := range promptAnimalLegs {
		add(fmt.Sprintf("How many legs does %s have?", a))
	}
	for _, w := range promptNouns {
		add(fmt.Sprintf("What is the plural of %q?", w))
	}
	for _, q := range promptMisc {
		add(q)
	}
	for _, c := range promptCountries {
		add(fmt.Sprintf("What continent is %s on?", c))
	}
	for _, con := range promptConcepts {
		add(fmt.Sprintf("Why is %s useful? Answer in one sentence.", con))
	}

	if n > len(out) {
		n = len(out)
	}
	return out[:n]
}
