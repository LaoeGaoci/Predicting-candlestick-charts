import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import baostock as bs

# Define the directory path
save_dir = '../image/'
os.makedirs(save_dir, exist_ok=True)


# 生成交易信号
def generate_signals(predictions, data):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0
    predictions = predictions.flatten()  # 将 predictions 转换为一维数组
    # .iloc 是一个整数位置（integer-location based）的索引器，用于通过位置选择数据，而不是通过标签。
    # [1:] 是一个切片操作，表示从索引 1 开始（包括索引 1）到最后一个元素的所有元素。
    signals['Signal'].iloc[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1)
    return signals['Signal'].values  # 返回一维数组


# 回测函数
def backtest_strategy(signals, data):
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    positions['Position'] = signals

    # 计算对数回报率
    data['close'] = data['Actual Close'].astype(float)  # 确保 'close' 列为浮点数
    data['LogReturn'] = np.log(data['close'] / data['close'].shift(1))

    # 移除可能引入的 NaN 值
    data = data.dropna(subset=['LogReturn'])

    # 根据持仓和对数回报率计算投资组合回报
    portfolio = positions['Position'] * data['LogReturn']

    # 计算累积对数回报率
    cumulative_log_returns = portfolio.cumsum()

    # 转换为累积回报
    cumulative_returns = np.exp(cumulative_log_returns)

    return cumulative_returns.values  # 返回一维数组


# 绩效评估
def evaluate_strategy(returns):
    returns = returns[~np.isnan(returns)]  # 移除 NaN 值
    if len(returns) == 0:
        return np.nan, np.nan, np.nan, np.nan

    total_return = returns[-1] - 1
    annualized_return = (returns[-1] ** (252 / len(returns))) - 1

    # 计算每日收益率
    daily_returns = np.diff(returns) / returns[:-1]
    if daily_returns.std() == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    max_drawdown = (returns / np.maximum.accumulate(returns) - 1).min()
    return total_return, annualized_return, sharpe_ratio, max_drawdown


# 参数

# 数据集划分比例
window_size = 60
train = 0.7
validate = 0.2
test = 1 - train - validate

# 训练参数
loss_function = 'mean_absolute_error'
epoch_count = 18
optimizer = "adam"
batch_size = 32
# %%
# 加载原始数据
lg = bs.login()
k_data = bs.query_history_k_data_plus(
    "sz.000001",  # 平安银行股票代码
    "date,open,high,low,close,volume",  # 获取的属性
    start_date="2018-1-01",  # 起始日期
    end_date="2023-12-31",  # 结束日期
    frequency="d"  # 按天获取
)
df = k_data.get_data()

# 数据预处理
df['Actual Close'] = pd.to_numeric(df['close'], errors='coerce')  # 将收盘价转化为数字
df['date'] = pd.to_datetime(df['date'])  # 将日期转化为可用格式
df = df.sort_values(by='date')  # 按日期排序
data = df[['date', 'Actual Close']]  # 提取日期与收盘价
data.set_index('date', inplace=True)  # 将收盘价作为索引

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data.values).reshape(1457)

# 将数据整合为所需格式，即每 60 天的数据为一个输入
xs = np.lib.stride_tricks.sliding_window_view(dataset, window_size)[:-1]
ys = dataset[window_size:]
assert len(xs) == len(ys)

# 划分训练集、验证集和测试集
count = len(xs)
train_count = int(train * count)
validate_count = int((validate + train) * count)

(train_x, validate_x, test_x) = np.array_split(xs, [train_count, validate_count])
(train_y, validate_y, test_y) = np.array_split(ys, [train_count, validate_count])

# 保存测试集原始数据
test_data = data[validate_count + window_size:].copy()
# %%
# 创建和拟合LSTM网络
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss=loss_function, optimizer=optimizer)
model.fit(train_x, train_y, epochs=epoch_count, batch_size=batch_size, verbose=2,
          validation_data=(validate_x, validate_y))
# %%
# 使用测试集进行测试
predict_close = model.predict(test_x)
predict_close = scaler.inverse_transform(predict_close)
test_data['Predicted Close'] = predict_close

# 计算误差
re = test_data['Predicted Close'] - test_data['Actual Close']
relative_re = re / test_data['Actual Close']
mse = re.pow(2).mean()
rmse = np.sqrt(mse)
std_re = np.std(re)

relative_mse = relative_re.pow(2).mean()
relative_rmse = np.sqrt(relative_mse)

print("MSE: ", mse)
print("RMSE: ", rmse)
print("Residual STD: ", std_re)
print("Relative MSE: ", relative_mse)
print("Relative RMSE: ", relative_rmse)

# 可视化预测结果
fig, ax = plt.subplots(dpi=240)
ax.plot(test_data.index, test_data['Actual Close'], label="Actual Close", color='blue')
ax.plot(test_data.index, test_data['Predicted Close'], label="Predicted Close", color='red')

# 将交易信号绘制成小柱状图
LSTM_prediction = generate_signals(predict_close, test_data)
bar_width = 0.3  # 设置柱状图的宽度
up_signal = np.where(LSTM_prediction == 1)[0]
down_signal = np.where(LSTM_prediction == -1)[0]

ax.bar(test_data.index[up_signal], test_data['Actual Close'].iloc[up_signal], color='green', width=bar_width,
       label='Buy Signal', alpha=0.6)
ax.bar(test_data.index[down_signal], test_data['Actual Close'].iloc[down_signal], color='red', width=bar_width,
       label='Sell Signal', alpha=0.6)

plt.ylabel("Close")
plt.xlabel("Date")
plt.title("Actual Close vs Predicted Close with Trading Signals")
plt.legend()
plt.savefig(save_dir + "Test_with_Signals.png")

fig = plt.figure(dpi=240)
ax_test = fig.add_subplot(111)
ax_test.hist(re)
ax_test.set_xlabel("Residual")
ax_test.set_title("Residual Distribution")
plt.savefig(save_dir + "Residual.png")

fig = plt.figure(dpi=240)
ax_test = fig.add_subplot(111)
ax_test.hist(relative_re)
ax_test.set_title("Relative Residual Distribution")
plt.savefig(save_dir + "RResidual.png")

# %%
# 调用交易策略和回测函数
LSTM_return = backtest_strategy(LSTM_prediction, test_data)

# 调用评估函数
LSTM_total_return, LSTM_annualized_return, LSTM_sharpe_ratio, LSTM_max_drawdown = evaluate_strategy(LSTM_return)

print('Total Return: ', LSTM_total_return)
print('Annualized Return: ', LSTM_annualized_return)
print('Sharpe ratio: ', LSTM_sharpe_ratio)
print('Max Drawdown: ', LSTM_max_drawdown)

bs.logout()
