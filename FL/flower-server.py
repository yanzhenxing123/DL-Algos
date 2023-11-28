"""
@Time : 2023/11/28 11:34
@Author : yanzx
@Description : 
"""

import flwr as fl

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))




