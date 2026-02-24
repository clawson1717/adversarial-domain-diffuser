import logging
import argparse
from src.evaluation import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Adversarial Domain Diffuser CLI")
    parser.add_argument("--eval", action="store_true", help="Run domain-specific evaluation")
    parser.add_argument("--domain", type=str, help="Specific domain to evaluate")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    print("Adversarial-Domain-Diffuser initialized.")
    
    if args.eval:
        evaluator = Evaluator()
        benchmarks = evaluator.load_benchmarks()
        
        if args.domain:
            if args.domain in benchmarks:
                evaluator.run_evaluation(args.domain, benchmarks[args.domain])
            else:
                print(f"Domain {args.domain} not found in benchmarks.")
        else:
            for domain, items in benchmarks.items():
                evaluator.run_evaluation(domain, items)

if __name__ == "__main__":
    main()
