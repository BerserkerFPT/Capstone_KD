"""
Experiment tracking and configuration saving module
Lưu lại các thí nghiệm, config, và kết quả để dễ dàng theo dõi và tái tạo
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import Config


class ExperimentTracker:
    """Class để theo dõi và lưu trữ thông tin experiments"""
    
    def __init__(self, experiments_dir="experiments"):
        """
        Initialize experiment tracker
        
        Args:
            experiments_dir: Thư mục lưu trữ thông tin experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # File lưu trữ tất cả experiments
        self.experiments_log = self.experiments_dir / "experiments_log.json"
        self.experiments_summary = self.experiments_dir / "experiments_summary.csv"
        
        # Load existing experiments
        self.experiments = self._load_experiments()
    
    def _load_experiments(self):
        """Load danh sách experiments đã lưu"""
        if self.experiments_log.exists():
            with open(self.experiments_log, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_experiments(self):
        """Save danh sách experiments"""
        with open(self.experiments_log, 'w', encoding='utf-8') as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False)
    
    def get_next_experiment_number(self):
        """Lấy số thứ tự experiment tiếp theo"""
        if not self.experiments:
            return 1
        return max(exp['experiment_number'] for exp in self.experiments) + 1
    
    def save_experiment_config(self, config_dict=None, experiment_name=None):
        """
        Lưu config của experiment hiện tại
        
        Args:
            config_dict: Dictionary chứa config (nếu None, sẽ lấy từ Config class)
            experiment_name: Tên experiment (nếu None, sẽ dùng Config.EXPERIMENT_NAME)
        
        Returns:
            experiment_number: Số thứ tự của experiment
        """
        experiment_number = self.get_next_experiment_number()
        
        if experiment_name is None:
            experiment_name = Config.EXPERIMENT_NAME
        
        # Lấy config từ Config class nếu không được cung cấp
        if config_dict is None:
            config_dict = self._extract_config_from_class()
        
        # Tạo experiment record
        experiment = {
            'experiment_number': experiment_number,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': config_dict,
            'models': config_dict.get('MODELS', []),
            'status': 'running',
            'results': {}
        }
        
        # Lưu vào danh sách
        self.experiments.append(experiment)
        self._save_experiments()
        
        # Lưu config riêng cho experiment này
        exp_dir = self.experiments_dir / f"exp_{experiment_number:03d}_{experiment_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = exp_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"📝 Experiment #{experiment_number} - '{experiment_name}' đã được lưu")
        print(f"{'='*70}")
        print(f"Config saved to: {config_file}")
        
        return experiment_number
    
    def update_experiment_results(self, experiment_number, model_name, results):
        """
        Cập nhật kết quả cho một model trong experiment
        
        Args:
            experiment_number: Số thứ tự experiment
            model_name: Tên model
            results: Dictionary chứa kết quả (tất cả strategies)
        """
        # Tìm experiment
        for exp in self.experiments:
            if exp['experiment_number'] == experiment_number:
                if 'results' not in exp:
                    exp['results'] = {}
                
                exp['results'][model_name] = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    **results
                }
                
                # Lưu lại
                self._save_experiments()
                
                # Lưu results riêng
                exp_dir = self.experiments_dir / f"exp_{experiment_number:03d}_{exp['experiment_name']}"
                results_file = exp_dir / f"results_{model_name}.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✓ Results for {model_name} saved to experiment #{experiment_number}")
                return
        
        print(f"⚠ Experiment #{experiment_number} not found")
    
    def mark_experiment_completed(self, experiment_number):
        """Đánh dấu experiment đã hoàn thành"""
        for exp in self.experiments:
            if exp['experiment_number'] == experiment_number:
                exp['status'] = 'completed'
                exp['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self._save_experiments()
                print(f"✓ Experiment #{experiment_number} marked as completed")
                return
        
        print(f"⚠ Experiment #{experiment_number} not found")
    
    def mark_experiment_failed(self, experiment_number, error_message=""):
        """Đánh dấu experiment bị lỗi"""
        for exp in self.experiments:
            if exp['experiment_number'] == experiment_number:
                exp['status'] = 'failed'
                exp['error'] = error_message
                exp['failed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self._save_experiments()
                print(f"✗ Experiment #{experiment_number} marked as failed")
                return
        
        print(f"⚠ Experiment #{experiment_number} not found")
    
    def generate_summary_table(self):
        """Tạo bảng tổng hợp tất cả experiments"""
        if not self.experiments:
            print("No experiments found")
            return None
        
        summary_data = []
        
        for exp in self.experiments:
            base_info = {
                'Exp#': exp['experiment_number'],
                'Name': exp['experiment_name'],
                'Date': exp['timestamp'],
                'Status': exp['status'],
                'Models': ', '.join(exp.get('models', [])),
                'Batch Size': exp['config'].get('BATCH_SIZE', 'N/A'),
                'Epochs': exp['config'].get('NUM_EPOCHS', 'N/A'),
                'LR': exp['config'].get('LEARNING_RATE', 'N/A'),
                'Dropout': exp['config'].get('DROPOUT_RATE', 'N/A'),
            }
            
            # Thêm kết quả của từng model
            results = exp.get('results', {})
            if results:
                for model_name, model_results in results.items():
                    row = base_info.copy()
                    row['Model'] = model_name
                    
                    # Lấy summary nếu có
                    if 'summary' in model_results:
                        summary = model_results['summary']
                        row['Best Strategy'] = summary.get('best_strategy', 'N/A')
                        row['Test Acc'] = summary.get('best_test_acc', 'N/A')
                        row['Test Loss'] = summary.get('best_test_loss', 'N/A')
                    else:
                        row['Test Acc'] = model_results.get('test_acc', 'N/A')
                        row['Test Loss'] = model_results.get('test_loss', 'N/A')
                    
                    summary_data.append(row)
            else:
                summary_data.append(base_info)
        
        df = pd.DataFrame(summary_data)
        
        # Lưu ra CSV
        df.to_csv(self.experiments_summary, index=False)
        print(f"\n📊 Summary table saved to: {self.experiments_summary}")
        
        return df
    
    def print_experiment_info(self, experiment_number):
        """In thông tin chi tiết của một experiment"""
        for exp in self.experiments:
            if exp['experiment_number'] == experiment_number:
                print(f"\n{'='*70}")
                print(f"EXPERIMENT #{experiment_number} - {exp['experiment_name']}")
                print(f"{'='*70}")
                print(f"Status: {exp['status']}")
                print(f"Created: {exp['timestamp']}")
                
                print(f"\n📋 Configuration:")
                for key, value in exp['config'].items():
                    print(f"  {key}: {value}")
                
                print(f"\n🤖 Models: {', '.join(exp.get('models', []))}")
                
                if exp.get('results'):
                    print(f"\n📊 Results:")
                    for model_name, model_results in exp['results'].items():
                        print(f"\n  {model_name}:")
                        
                        # In summary trước
                        if 'summary' in model_results:
                            print(f"    📌 Summary:")
                            for metric, value in model_results['summary'].items():
                                print(f"       {metric}: {value}")
                        
                        # In tất cả strategies
                        print(f"    📊 All Strategies:")
                        for strategy_name, strategy_data in model_results.items():
                            if strategy_name not in ['timestamp', 'summary']:
                                print(f"       {strategy_name}:")
                                if isinstance(strategy_data, dict):
                                    for metric, value in strategy_data.items():
                                        print(f"          {metric}: {value}")
                
                print(f"{'='*70}\n")
                return
        
        print(f"⚠ Experiment #{experiment_number} not found")
    
    def list_all_experiments(self):
        """Liệt kê tất cả experiments"""
        if not self.experiments:
            print("No experiments found")
            return
        
        print(f"\n{'='*70}")
        print(f"ALL EXPERIMENTS")
        print(f"{'='*70}")
        
        for exp in self.experiments:
            status_icon = {
                'running': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(exp['status'], '❓')
            
            print(f"\n{status_icon} Exp #{exp['experiment_number']}: {exp['experiment_name']}")
            print(f"   Date: {exp['timestamp']}")
            print(f"   Status: {exp['status']}")
            print(f"   Models: {', '.join(exp.get('models', []))}")
            
            if exp.get('results'):
                print(f"   Results: {len(exp['results'])} model(s) completed")
        
        print(f"\n{'='*70}\n")
    
    def compare_experiments(self, exp_numbers):
        """
        So sánh nhiều experiments
        
        Args:
            exp_numbers: List các số thứ tự experiments cần so sánh
        """
        print("\n" + "="*100)
        print(f"COMPARING EXPERIMENTS: {', '.join(f'#{n}' for n in exp_numbers)}")
        print("="*100)
        
        comparison_data = []
        
        for exp in self.experiments:
            if exp['experiment_number'] in exp_numbers:
                exp_num = exp['experiment_number']
                exp_name = exp['experiment_name']
                
                # Lấy config chính
                config = exp['config']
                
                # Lấy kết quả của từng model
                results = exp.get('results', {})
                
                for model_name, model_results in results.items():
                    # Nếu có summary, dùng summary
                    if 'summary' in model_results:
                        summary = model_results['summary']
                        row = {
                            'Exp#': exp_num,
                            'Exp Name': exp_name,
                            'Model': model_name,
                            'Batch Size': config.get('BATCH_SIZE', 'N/A'),
                            'Epochs': config.get('NUM_EPOCHS', 'N/A'),
                            'LR': config.get('LEARNING_RATE', 'N/A'),
                            'Best Strategy': summary.get('best_strategy', 'N/A'),
                            'Test Acc': summary.get('best_test_acc', 'N/A'),
                            'Test Loss': summary.get('best_test_loss', 'N/A'),
                        }
                        comparison_data.append(row)
                        
                        # Hiển thị tất cả strategies
                        print(f"\n📊 Exp #{exp_num} - {model_name}:")
                        print(f"   Config: BS={config.get('BATCH_SIZE')}, "
                              f"Epochs={config.get('NUM_EPOCHS')}, "
                              f"LR={config.get('LEARNING_RATE')}")
                        
                        for strategy_name, strategy_results in model_results.items():
                            if strategy_name not in ['timestamp', 'summary'] and isinstance(strategy_results, dict):
                                acc = strategy_results.get('accuracy', 'N/A')
                                loss = strategy_results.get('test_loss', 'N/A')
                                print(f"   - {strategy_name:20s}: Acc={acc:>6}%, Loss={loss}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print("\n" + "="*100)
            print("COMPARISON TABLE")
            print("="*100)
            print(df.to_string(index=False))
            print("="*100)
            return df
        else:
            print("\n⚠ No data found for comparison")
            return None
    
    def export_experiment(self, experiment_number, output_file):
        """
        Export một experiment ra file JSON riêng
        
        Args:
            experiment_number: Số thứ tự experiment
            output_file: Đường dẫn file output
        """
        for exp in self.experiments:
            if exp['experiment_number'] == experiment_number:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(exp, f, indent=2, ensure_ascii=False)
                print(f"✓ Experiment #{experiment_number} exported to: {output_file}")
                return
        
        print(f"⚠ Experiment #{experiment_number} not found")
    
    def get_best_model_per_experiment(self):
        """Lấy model tốt nhất của mỗi experiment"""
        best_models = []
        
        for exp in self.experiments:
            exp_num = exp['experiment_number']
            exp_name = exp['experiment_name']
            results = exp.get('results', {})
            
            best_acc = 0
            best_model = None
            
            for model_name, model_results in results.items():
                if 'summary' in model_results:
                    acc = model_results['summary'].get('best_test_acc', 0)
                    if isinstance(acc, (int, float)) and acc > best_acc:
                        best_acc = acc
                        best_model = model_name
            
            if best_model:
                best_models.append({
                    'Exp#': exp_num,
                    'Exp Name': exp_name,
                    'Best Model': best_model,
                    'Test Acc': best_acc
                })
        
        if best_models:
            df = pd.DataFrame(best_models)
            print("\n" + "="*70)
            print("BEST MODEL PER EXPERIMENT")
            print("="*70)
            print(df.to_string(index=False))
            print("="*70)
            return df
        
        return None
    
    def _extract_config_from_class(self):
        """Trích xuất config từ Config class"""
        config_dict = {}
        
        # Lấy tất cả attributes của Config class
        for attr in dir(Config):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(Config, attr)
                # Chỉ lưu các giá trị cơ bản (không lưu methods)
                if not callable(value):
                    config_dict[attr] = value
        
        return config_dict


# ============================================================
# Helper Functions
# ============================================================

def save_current_experiment(experiment_name=None):
    """
    Helper function để lưu experiment hiện tại
    
    Args:
        experiment_name: Tên experiment (nếu None, dùng Config.EXPERIMENT_NAME)
    
    Returns:
        experiment_number: Số thứ tự experiment
    """
    tracker = ExperimentTracker()
    return tracker.save_experiment_config(experiment_name=experiment_name)


def update_experiment_results(experiment_number, model_name, results):
    """
    Helper function để cập nhật kết quả
    
    Args:
        experiment_number: Số thứ tự experiment
        model_name: Tên model
        results: Dictionary chứa kết quả
    """
    tracker = ExperimentTracker()
    tracker.update_experiment_results(experiment_number, model_name, results)


def mark_completed(experiment_number):
    """Helper function để đánh dấu experiment hoàn thành"""
    tracker = ExperimentTracker()
    tracker.mark_experiment_completed(experiment_number)


def mark_failed(experiment_number, error_message=""):
    """Helper function để đánh dấu experiment thất bại"""
    tracker = ExperimentTracker()
    tracker.mark_experiment_failed(experiment_number, error_message)


# ============================================================
# Main - Demo and CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Experiment Tracker - View and manage experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python save.py                          # List all experiments
  python save.py --exp 1                  # View experiment #1 details
  python save.py --summary                # Generate summary table
  python save.py --compare 1 2 3          # Compare experiments 1, 2, 3
  python save.py --export 1 exp1.json     # Export experiment #1
  python save.py --best                   # Show best model per experiment
        """
    )
    
    parser.add_argument('--exp', type=int, help='View details of specific experiment number')
    parser.add_argument('--summary', action='store_true', help='Generate summary table')
    parser.add_argument('--compare', nargs='+', type=int, help='Compare multiple experiments')
    parser.add_argument('--export', nargs=2, metavar=('EXP_NUM', 'FILE'), help='Export experiment to JSON')
    parser.add_argument('--best', action='store_true', help='Show best model per experiment')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Execute command
    if args.exp:
        tracker.print_experiment_info(args.exp)
    elif args.summary:
        df = tracker.generate_summary_table()
        if df is not None:
            print("\n📊 Summary Table:")
            print(df.to_string(index=False))
    elif args.compare:
        tracker.compare_experiments(args.compare)
    elif args.export:
        exp_num = int(args.export[0])
        output_file = args.export[1]
        tracker.export_experiment(exp_num, output_file)
    elif args.best:
        tracker.get_best_model_per_experiment()
    else:
        # Default: list all experiments
        tracker.list_all_experiments()
