def get_dict(element):
    import re
    l= {}
    res = re.findall('[A-Z][^A-Z]*', element)
    for element in res:
        if '.' in element :
            l[re.findall("(\d+\.\d+|[A-Za-z]+)", element)[0]] = float(re.findall('(\d+\.\d+|[A-Za-z]+)', element )[1])
        if len(re.findall('(\d+|[A-Za-z]+)', element )) == 2 :
            l[re.findall('(\d+|[A-Za-z]+)', element )[0]] = float(re.findall('(\d+|[A-Za-z]+)', element )[1])
        if len(re.findall('(\d+|[A-Za-z]+)', element )) == 1 :
            l[re.findall('(\d+|[A-Za-z]+)', element )[0]] = 1     
    return l
def run_herows():
	global elements
	import pandas as pd
	elements = pd.read_excel("Material Database finale.xlsx")



def fCO2_max(alloy):
    fCO2_max = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['CO2 footprint max (CO2/mol)'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            fCO2_max = fCO2_max + ri*ci
    return(fCO2_max)

def fE_min(alloy):
    fE_min = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['Energy footprint min (MJ/mol)'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            fE_min = fE_min + ri*ci
    return(fE_min)

def fE_max(alloy):
    fE_max = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['Energy footprint max (MJ/mol)'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            fE_max = fE_max + ri*ci
    return(fE_max)

def hhi(alloy):
    hhi = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['HHI'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            hhi = hhi + ri*ci
    return(hhi)

def esg(alloy):
    esg = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['ESG Score'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            esg = esg + ri*ci
    return(esg)

def Supply_risk(alloy):
    sr = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['Supply risk'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            sr = sr + ri*ci
    return(sr)

def c_max(alloy):
    c_max = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    l = []
    for element in propotion_dict.keys():
        l.append(elements[elements['Elements '] == element ]['Companionality'].reset_index(drop=True)[0])
    c_max = max(l)
    return(c_max)

def p_max(alloy):
    p_max = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    l = []
    for element in propotion_dict.keys():
        l.append(elements[elements['Elements '] == element ]['World production (tons per year)'].reset_index(drop=True)[0])
    p_max = max(l)
    return(p_max)

def r_max(alloy):
    r_max = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    l = []
    for element in propotion_dict.keys():
        l.append(elements[elements['Elements '] == element ]['World reserve (tons)'].reset_index(drop=True)[0])
    r_max = max(l)
    return(r_max)

def c_avg(alloy):
    c_avg = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['Companionality'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            c_avg = c_avg + ri*ci
    return(c_avg)

def p_avg(alloy,N):
    p_avg = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['World production (tons per year)'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            p_avg = p_avg + (1/ri)*ci
    return(p_avg*N)

def r_avg(alloy,N):
    r_avg = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['World reserve (tons)'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            r_avg = r_avg + (1/ri)*ci
    return(r_avg*N)

def fCO2(alloy):
    fCO2 = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['CO2 footprint  (CO2/mol)'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            fCO2 = fCO2 + ri*ci
    return(fCO2)

def fE(alloy):
    fE = 0
    propotion_dict = get_dict(alloy)
    somme_stch = sum(propotion_dict.values())
    for element in propotion_dict.keys():
            ri = elements[elements['Elements '] == element ]['Energy footprint (MJ/mol)'].reset_index(drop=True)[0]
            ci = propotion_dict[element]/somme_stch
            fE = fE + ri*ci
    return(fE)

def assign_decile(value,col,df):
    deciles = df[col].quantile(q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for i, decile_value in enumerate(deciles):
        if value <= decile_value:
            return i + 1  # Les déciles sont numérotés de 1 à 10
    return 10


def bijection_decile_value(n) :
     l = {1:"<10%" , 
          2:str('10% - 20%')  ,
          3:str('20% - 30%'),
          4:str('30% - 40%'),
          5:str('40% - 50%'),
          6:str('50% - 60%'),
          7:str('60% - 70%'),
          8:str('70% - 80%'),
          9:str('80% - 90%'),
          10:"<90%"}
     return (l[n])

def bijection_varibles(str) :
     l = {"Carbon Footprint (g CO2/mol)" : "FCO2",  
            "Energy Footprint (MJ/mol)" : "FE", 
            "Herfindahl-Hirschman Index" : "hhi",   
            "ESG risk" : "esg", 
            "Supply risk" : "sr", 
            "Average Companionality" : "C_avg",
            "Maximal Companionality" : "C_max", 
            "Average Production (t/year)" : "P_avg" ,
            "Maximal Production (t/year)" : "P_max",
            "Average Reserves (tons)" : "R_avg",
            "Maximal Reserves (tons)" : "R_max"
            }
     return(l[str])



  
