#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'David Zhang'

from thrift import Thrift
from thrift.transport import TSocket, TTransport
from thrift.protocol import TCompactProtocol
from thrift.server import TServer
from rpc import MailQueryService, ttypes, HOST, PORT
from spam.impl import *
from util import bytes2str
import logging


class MailQueryServiceHandler(MailQueryService.Iface):

    model = None

    def __init__(self, model: BayesSpamModel):
        if model is None:
            raise TypeError('model can not be null...')
        self.model = model

    def queryMailClass(self, data):
        # data是字节数组
        try:
            # 注意编码问题
            clazz = self.model.predict_one_bytes(data)
            logging.info("predict class: %s" % clazz)
            return ttypes.RpcResult(0, clazz)
        except Exception as e:
            logging.exception(repr(e))
            return ttypes.RpcResult(1, -1)


def run(host=HOST, port=PORT):

    try:
        # 注意确保模型参数文件存在，若不存在则先运行examples.py中的train方法
        model = BayesSpamModel.import_model()

        # 线程池
        server = TServer.TThreadPoolServer(
            MailQueryService.Processor(MailQueryServiceHandler(model)),
            # 注意py3的socket tcp通信默认采用tcpv6
            TSocket.TServerSocket(host, port),
            TTransport.TFramedTransportFactory(),
            TCompactProtocol.TCompactProtocolFactory()
        )

        server.setNumThreads(3)
        server.serve()

    except Thrift.TException as e:
        logging.exception(repr(e))


if __name__ == '__main__':
    run()