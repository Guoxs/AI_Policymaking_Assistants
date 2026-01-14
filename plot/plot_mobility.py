### 绘制某个州的具体政策（policy) 即为inflow 的控制
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
# ---------- 全局字体 ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})


STATE_ABBR_5 = {
    "arizona": "AZ",
    "mississippi": "MS",
    "new mexico": "NM",
    "texas": "TX",
    "virginia": "VA",
}

def get_inflow_df(path, state_i):
    with open(path, 'rb') as f:
        policy  = pickle.load(f)
    state_list = ['arizona','mississippi', 'new mexico', 'texas', 'virginia']
    state_list.remove(state_i)
    inflow_dict = {state: [] for state in state_list}
    for i in range(len(policy)):
        for state in state_list:
            if state == state_i:
                continue
            inflow = policy[i].outflow[state]
            inflow_dict[state].append(inflow)
    inflow_df = pd.DataFrame(inflow_dict)
    return inflow_df

def main():
    state_i = 'mississippi'
    state_list = ['arizona','new mexico', 'texas', 'virginia']
    for test_state in state_list:
    # test_state = 'virginia'
        #####只绘制第二个time period的数据
        plt.figure(figsize=(8,6))
        base_path = f'outputs\\5 states\\6 weeks\\no_action_2025-12-17-20-07-44\mobility\\{state_i}_mobility.pkl'
        base_inflow_df = get_inflow_df(base_path, state_i)
        base_inflow_df[test_state].plot(color= '#5d8eb9', label='Base', marker = 'o')
        gpt41_mini_path = f"outputs\\5 states\\6 weeks\\gpt-4.1-mini\gpt-4.1-mini_2025-12-19-22-04-40\\mobility\{state_i}_mobility.pkl"
        gpt41_mini_inflow_df = get_inflow_df(gpt41_mini_path, state_i)
        gpt41_mini_inflow_df[test_state].plot(color='orange', label='LLM Agent',marker='s', linestyle='--')
        # 每隔42个点在图中mark一个点
        interval = 42
        x_vals = gpt41_mini_inflow_df.index[::interval]
        y_vals = gpt41_mini_inflow_df[test_state].iloc[::interval]

        start_date = pd.Timestamp('2020-04-12') 
        T = len(gpt41_mini_inflow_df)
        x = pd.date_range(start=start_date, periods=T, freq='D')

        plt.xticks(ticks=range(0, len(x), 14), labels=x[::14].strftime('%Y-%m-%d'), fontsize=20)
        plt.yticks(fontsize=20)
        plt.scatter(x_vals, y_vals, color='orange', marker='s', s=40)


        for i in range(len(x_vals)):
            plt.axvline(x=x_vals[i], color='gray', linestyle='--', alpha=0.5)
            if i < len(x_vals) - 1:
                plt.axvspan(x_vals[i], x_vals[i+1], color=('lightblue' if i % 2 == 0 else 'lightyellow'), alpha=0.2)
            # if i == len(x_vals) - 1:
            #     plt.axvspan(x_vals[i], T, color=('lightblue' if i % 2 == 0 else 'lightyellow'), alpha=0.2)
        plt.xlim(interval, interval*2)
        if test_state == 'new mexico':
            plt.legend(fontsize=25)
        # plt.xlabel('Days', fontsize=24)
        plt.ylabel(f'Inflow to {STATE_ABBR_5[state_i]}', fontsize=30)

        confirm_df = pd.read_csv(f'outputs\\5 states\\6 weeks\\gpt-4.1-mini\\gpt-4.1-mini_2025-12-19-22-04-40\\results\\{test_state}_results.csv')
        Q_gt = confirm_df['Q_gt'].to_numpy(dtype=float)
        Q_agent = confirm_df['Q_pred'].to_numpy(dtype=float)

        # ax1 = plt.gca()
        # ax2 = ax1.twinx()
        # ax2.plot(Q_gt, color='black', label='Q_gt', linestyle='--')
        # ax2.plot(Q_agent, color='blue', label='Q_agent', linestyle='-.')
        # ax2.set_ylabel('Confirmed Cases', fontsize=24)
        # ax2.tick_params(axis='y', labelsize=20)
        # ax2.legend(loc='upper right', fontsize=16)

        plt.title(f'{STATE_ABBR_5[test_state]}', fontsize=35)
        plt.tight_layout()
        plt.savefig(f'figures/5 states/mississippi/{test_state}_mobility.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    main()