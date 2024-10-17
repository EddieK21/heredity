import csv
import itertools
import sys

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {
        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },
        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },
        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}

def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                         False if row["trait"] == "0" else None)
            }
    return data

def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def get_gene_count(person, one_gene, two_genes):
    """Helper function to determine how many copies of the gene a person has"""
    if person in two_genes:
        return 2
    elif person in one_gene:
        return 1
    return 0

def calculate_gene_probability(parent_genes):
    """Calculate probability of passing on the gene based on parent's genes"""
    if parent_genes == 0:
        return PROBS["mutation"]
    elif parent_genes == 1:
        return 0.5
    else:  # parent_genes == 2
        return 1 - PROBS["mutation"]

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.
    """
    joint_prob = 1

    for person in people:
        # Get person's gene count
        gene_count = get_gene_count(person, one_gene, two_genes)
        
        # Calculate probability of having this number of genes
        gene_prob = 1
        mother = people[person]["mother"]
        father = people[person]["father"]

        if mother is None and father is None:
            # No parents - use unconditional probability
            gene_prob = PROBS["gene"][gene_count]
        else:
            # Has parents - calculate inheritance probability
            mother_genes = get_gene_count(mother, one_gene, two_genes)
            father_genes = get_gene_count(father, one_gene, two_genes)

            # Calculate probability based on number of genes to inherit
            if gene_count == 0:
                # Need to NOT inherit from both parents
                prob_no_mother = 1 - calculate_gene_probability(mother_genes)
                prob_no_father = 1 - calculate_gene_probability(father_genes)
                gene_prob = prob_no_mother * prob_no_father
            elif gene_count == 1:
                # Need to inherit from exactly one parent
                prob_yes_mother = calculate_gene_probability(mother_genes)
                prob_no_mother = 1 - prob_yes_mother
                prob_yes_father = calculate_gene_probability(father_genes)
                prob_no_father = 1 - prob_yes_father
                gene_prob = (prob_yes_mother * prob_no_father + 
                           prob_no_mother * prob_yes_father)
            else:  # gene_count == 2
                # Need to inherit from both parents
                prob_yes_mother = calculate_gene_probability(mother_genes)
                prob_yes_father = calculate_gene_probability(father_genes)
                gene_prob = prob_yes_mother * prob_yes_father

        # Calculate trait probability
        trait_prob = PROBS["trait"][gene_count][person in have_trait]
        
        # Update joint probability
        joint_prob *= gene_prob * trait_prob

    return joint_prob

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    """
    for person in probabilities:
        # Update gene probability
        if person in two_genes:
            probabilities[person]["gene"][2] += p
        elif person in one_gene:
            probabilities[person]["gene"][1] += p
        else:
            probabilities[person]["gene"][0] += p
            
        # Update trait probability
        probabilities[person]["trait"][person in have_trait] += p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # Normalize gene probabilities
        gene_total = sum(probabilities[person]["gene"].values())
        if gene_total != 0:
            for gene in probabilities[person]["gene"]:
                probabilities[person]["gene"][gene] /= gene_total
        
        # Normalize trait probabilities
        trait_total = sum(probabilities[person]["trait"].values())
        if trait_total != 0:
            for trait in probabilities[person]["trait"]:
                probabilities[person]["trait"][trait] /= trait_total

def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):
        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")

if __name__ == "__main__":
    main()