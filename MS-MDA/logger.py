# import logging

# # 配置日志记录器
# logging.basicConfig(filename='new.log', level=logging.INFO)

# # 部分的print语句
# logging.info('这是要保存到日志文件的内容')

# # 其他的代码和print语句

# # 关闭日志记录器
# logging.shutdown()





import logging

# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# 创建日期和时间的格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建文件处理器
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(file_handler)

# 打印日志消息
logger.info('这是一条日志消息')