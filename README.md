# Shannon
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/fingpt.svg?color=brightgreen)
This project(Shannon) is an open-source framework designed to offer efficient backtesting capabilities for multi-factor stock selection strategies. The primary objective of this framework is to facilitate strategy researchers and quantitative investors in testing, optimizing, and implementing their stock selection strategies with ease.

# __Key Features__
Modular Design: The framework is architected to decouple modules like data preparation, strategy formulation, and performance evaluation comprehensively. Users can switch between various modules conveniently, thereby facilitating efficient strategy development.

Strategy-Centric: The focus is on writing the strategy, so users don't have to be distracted by data preparation or performance assessment tasks.

High Flexibility: It combines factor-based stock selection with various filters, offering users a high level of customization. Regardless of the strategy or factors you're based on, you can efficiently implement them within this framework. 

# __Install__

After downloading and extracting, switch to the project directory.
```shell
pip install -r requirements.txt
```

# __Steps__

1.In config.py, users can configure the data storage path, backtesting parameters, and more. 
2.Run preprocess.py for data preprocessing. 
3.Execute backtest.py to complete the backtesting and output the strategy evaluation metrics.
4.This framework includes a built-in sample strategy located in the strategies folder. Users can follow its pattern to craft their own strategies.

# __Data Api__
The data used by this framework includes individual stock trading data,index trading data and  financial data. This data needs to be retrieved via an API and then decompressed locally. Users can contact the author to obtain free API access permissions (each type of data can be downloaded once a week).