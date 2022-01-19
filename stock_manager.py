from pandas_datareader import data as web
import numpy as np
import json
import pandas as pd
import os
import numpy as np
import fundament as f
from scipy.stats import norm

class Cont:
    
    def liquidez_corrente(self, ativo_circulante, passivo_circulante):
        return ativo_circulante/passivo_circulante
    
    def liquidez_seca(self, ativo_circulante, passivo_circulante, estoque):
        return (ativo_circulante-estoque)/passivo_circulante
    
    def liquidez_geral(self, ativo_circulante, ativo_rlp, passivo_circulante, passivo_rlp):
        return (ativo_circulante + ativo_rlp)/(passivo_circulante + passivo_rlp)
    
    def giro_ativo(self, receita_operacional, ativo_med_total):
        return receita_operacional/ativo_med_total
    
    def giro_contas_a_receber(self, receita_operacional, contas_a_receber_media):
        return receita_operacional/contas_a_receber_media
    
    def giro_contas_a_pagar(self, compras_media, media_fornecedores):
        return compras_media/media_fornecedores
    
    def giro_de_estoques(self, cmv, estoque_médio):
        return cmv/estoque_médio
    
    def prazo_medio(self, x):
        return 365/x
    
    def ciclo_de_caixa(self, receita_operacional, contas_a_receber_media, compras_media, media_fornecedores, cmv, estoque_médio):
        return int(self.prazo_medio(self.giro_de_estoques(cmv, estoque_médio)) + self.prazo_medio(self.giro_contas_a_receber(receita_operacional, contas_a_receber_media)) - self.prazo_medio(self.giro_contas_a_pagar(compras_media, media_fornecedores)))
    
    def margem_bruta(self, lucro_bruto, receita_líquida):
        return lucro_bruto/receita_líquida
    
    def margem_operacional(self, lucro_operacional, receita_líquida):
        return lucro_operacional/receita_líquida
    
    def margem_liquida(self, lucro_liquido, receita_líquida):
        return lucro_liquido/receita_líquida
    
    def roa(self, lucro_operacional, impostos, ativo_médio):
        return (lucro_operacional - impostos)/ ativo_médio
    
    def roe(self, lucro_liquido, pl_médio):
        return lucro_liquido/pl
    
    def endividamento_financeiro(self, emprestimos_cp, emprestimos_lp, debentures, dividas, pl):
        return (emprestimos_cp + emprestimos_lp + debentures)/(dividas + pl)
    
    def capital_proprio(emprestimos_cp, emprestimos_lp, debentures, dividas, pl):
        return (1 - endividamento_financeiro(emprestimos_cp, emprestimos_lp, debentures, dividas, pl))
    
    def endividamento_cp(self, endividamento_cp, endividamento_total):
        return endividamento_cp/endividamento_total
    
class Price:
    
    def __init__(self, symbols, start, end, adjust_price):
        self.symbols = symbols
        self.start = start
        self.end = end
        self.adjust_price = adjust_price
        self.historical_prices = web.get_data_yahoo( self.symbols,
                                                 self.start,
                                                 self.end,
                                                 self.adjust_price )
        self.high = self.historical_prices["High"]
        self.low = self.historical_prices["Low"]
        self.open = self.historical_prices["Open"]
        self.close = self.historical_prices["Close"]
        self.volume = self.historical_prices["Volume"]
        self.adjclose = self.historical_prices["Adj Close"]

class Operação:
    
    def __init__(self, preco, qtd, data, ticker, tipo):
        self.preco_de_compra = preco
        self.qtd = qtd
        self.data_de_compra = data
        self.tipo = tipo
        if self.tipo =="V":
            self.qtd = self.qtd * -1
        self.dict = {'Ticker': ticker, 'Price': preco , 'Quantidade': self.qtd , 'Data': data, 'Tipo': tipo, 'Financeiro': preco * self.qtd}
        self.df = pd.DataFrame.from_dict(self.dict, orient = "index").T
        self.save  = self.df.to_csv("Operações\\"+ticker+" - "+data )

class Portfolio:
    
    def consolidar(self):
        df = []
        for x in os.listdir("Operações")[1:]:
            data = pd.read_csv("Operações\\"+x)
            df.append(data)
        return pd.concat(df)
        
    def __init__(self):
        self.nav = "R$  " + str(self.consolidar()["Financeiro"].sum())
        
        
class Erk:
    
    def drawdown(self, return_series: pd.Series):
        """
        Takes a times series of assests returns
        Computes and returns a DataFrame that contains:
        the wealth index
        the previous peaks
        percent drawdowns
        """
        wealth_index = 1000*(1+return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks
        return pd.DataFrame({
            "Wealth":wealth_index,
            "Peaks":previous_peaks,
            "Drawdown": drawdowns})

    def skewness(self, r):
        """
        função alternativa a scipy.stats.skew()
        calcula o skewness de uma série ou dataframe
        retorna um float ou series
        """
        demeaned_r = r - r.mean()
        # use o desvio-padrão populacional
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()
        return exp/sigma_r**3

    def kourtosis(self, r):
        """
        função alternativa a scipy.stats.kourtosis()
        calcula o skewness de uma série ou dataframe
        retorna um float ou series
        """
        demeaned_r = r - r.mean()
        # use o desvio-padrão populacional
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()
        return exp/sigma_r**4
    
    def is_normal(self, r, level=0.01):
        """
        Aplica o teste jarque-bera para determinar se uma série é normalmente distribuida ou não
        teste aplicado com 99% de confiança
        Retorna verdadeiro caso seja normalmente distribuído, caso contrário falso
        """
        statistics, p_value = scipy.stats.jarque_bera(r)
        print(scipy.stats.jarque_bera(r))
        return p_value > level

    def semideviation(self, r):
        """
        Retorna o desvio dos valores negativos de uma série ou data frame
        """
        r = r.fillna(0)
        return r[r<0].std(ddof=0)

    def var_historic(self, r, level=5):
        """
        VaR histórico
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Valor esperado: DataFrame ou Series")
            
    def var_gaussian(self, r, level=5, modified=False):
        """
        Retorna o VaR gaussiano de uma série ou dataframe 
        """
        # cálculo do z-score
        z = norm.ppf(level/100)
        if modified:
            # modifica o z-score com base no kourtosis e skewness
            s = skewness(r)
            k = kourtosis(r)
            z = (z +
                    (z**2 - 1)*s/6 + 
                    (z**3 -3*z)*(k-3)/24 -
                    (2*z**3 - 5*z)*(s**2)/36
                )
        return -(r.mean() + z * r.std(ddof=0))
    
    def annualize_rets(self, r, periods_per_year):
        """
        retorna o retorno composto por perídos no ano de um dataframe
        """
        compounded_growth = (1+r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year/n_periods)-1

    def annualize_vol(self, r, periods_per_year):
        """
        Anualiza a volatilidade de retornos inferindo períodos por ano
        """
        return r.std()*(periods_per_year**0.5)


    
class Indic:
    
    def __init__(self, ticker):
        
        import fundament as f
        
        self.ticker = ticker
        self.indicators = pd.DataFrame.from_dict(f.get_data())[self.ticker]
        
        
        