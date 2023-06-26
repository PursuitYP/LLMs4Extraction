""" 沉积岩 @何忠谋 """
head = "head entity"
tail = "tail entity"
prompts = f'please tell me the relationship between {head} and {tail}, choose from RelatedTo,FormOf,IsA,PartOf,HasA,UsedFor,CapableOf,AtLocation,Causes,HasSubevent,HasFirstSubevent,HasLastSubevent,HasPrerequisite,HasProperty,MotivatedByGoal,ObstructedBy,Desires,CreatedBy,Synonym,Antonym,DistinctFrom,DerivedFrom,SymbolOf,DefinedAs,MannerOf,LocatedNear,HasContext,SimilarTo,EtymologicallyRelatedTo,EtymologicallyDerivedFrom,CausesDesire,MadeOf,ReceivesAction,ExternalURL'


from revChatGPT.V3 import Chatbot
import json

def ChatGPT_extract_entity_and_relation(paragraph: str):
    chatbot = Chatbot(api_key=API_KEY)
    
    system_prompt = "I want you to act as a entity and relation extractor to help me build an academic knowledge graph from several paragraphs."
    chatbot.ask(system_prompt)
    
    prompt1 = f"""
I will give you a paragraph. Extract as many named entities as possible from it. Your answer should only contain a list and nothing else. Here is an example:

paragraph: Scientists conducted depth first search into borealis and found that the formation of Magnolite is obstructed by solar cycle.

your answer: 
[
"depth first search",
"borealis",
"Magnolite",
"solar cycle",
]

Here is the paragraph you should process:
{paragraph}
"""
    entity_list = chatbot.ask(prompt1)
    print(entity_list)
    
    prompt2 = "This is the entity list you have just generated." + str(entity_list) + """

Classify every entity in into one of the categories in the following list. You should not classify any entity into a category that in not in the following list.

[human, humanAgriculture, humanCommerce, humanDecision, humanEnvirAssessment, humanEnvirConservation, humanEnvirControl, humanEnvirStandards, humanJurisdiction, humanKnowledgeDomain, humanResearch, humanTechReadiness, humanTransportation, matr, matrAerosol, matrAnimal, matrBiomass, matrElement, matrElementalMolecule, matrEnergy, matrEquipment, matrFacility, matrIndustrial, matrInstrument, matrIon, matrIsotope, matrMicrobiota, matrMineral, matrNaturalResource, matrOrganicCompound, matrParticle, matrPlant, matrRock, matrRockIgneous, matrSediment, matrWater, phen, phenAtmo, phenAtmoCloud, phenAtmoFog, phenAtmoFront, phenAtmoLightning, phenAtmoPrecipitation, phenAtmoPressure, phenAtmoTransport, phenAtmoWind, phenAtmoWindMesoscale, phenBiol, phenCryo, phenCycle, phenCycleMaterial, phenEcology, phenElecMag, phenEnergy, phenEnvirImpact, phenFluidDynamics, phenFluidInstability, phenFluidTransport, phenGeol, phenGeolFault, phenGeolGeomorphology, phenGeolSeismicity, phenGeolTectonic, phenGeolVolcano, phenHelio, phenHydro, phenMixing, phenOcean, phenOceanCoastal, phenOceanDynamics, phenPlanetClimate, phenReaction, phenSolid, phenStar, phenSystem, phenSystemComplexity, phenWave, phenWaveNoise, proc, procChemical, procPhysical, procStateChange, procWave, prop, propBinary, propCapacity, propCategorical, propCharge, propChemical, propConductivity, propCount, propDifference, propDiffusivity, propDimensionlessRatio, propEnergy, propEnergyFlux, propFraction, propFunction, propIndex, propMass, propMassFlux, propOrdinal, propPressure, propQuantity, propRotation, propSpace, propSpaceDirection, propSpaceDistance, propSpaceHeight, propSpaceLocation, propSpaceMultidimensional, propSpaceThickness, propSpeed, propTemperature, propTemperatureGradient, propTime, propTimeFrequency, realm, realmAstroBody, realmAstroHelio, realmAstroStar, realmAtmo, realmAtmoBoundaryLayer, realmAtmoWeather, realmBiolBiome, realmClimateZone, realmCryo, realmEarthReference, realmGeol, realmGeolBasin, realmGeolConstituent, realmGeolContinental, realmGeolOrogen, realmHydro, realmHydroBody, realmLandAeolian, realmLandCoastal, realmLandFluvial, realmLandGlacial, realmLandOrographic, realmLandProtected, realmLandTectonic, realmLandVolcanic, realmLandform, realmOcean, realmOceanFeature, realmOceanFloor, realmRegion, realmSoil, rela, relaChemical, relaClimate, relaHuman, relaMath, relaPhysical, relaProvenance, relaSci, relaSpace, relaTime, repr, reprDataFormat, reprDataModel, reprDataProduct, reprDataService, reprDataServiceAnalysis, reprDataServiceGeospatial, reprDataServiceReduction, reprDataServiceValidation, reprMath, reprMathFunction, reprMathFunctionOrthogonal, reprMathGraph, reprMathOperation, reprMathSolution, reprMathStatistics, reprSciComponent, reprSciFunction, reprSciLaw, reprSciMethodology, reprSciModel, reprSciProvenance, reprSciUnits, reprSpace, reprSpaceCoordinate, reprSpaceDirection, reprSpaceGeometry, reprSpaceGeometry3D, reprSpaceReferenceSystem, reprTime, reprTimeDay, reprTimeSeason, state, stateBiological, stateChemical, stateDataProcessing, stateEnergyFlux, stateFluid, stateOrdinal, statePhysical, stateRealm, stateRole, stateRoleBiological, stateRoleChemical, stateRoleGeographic, stateRoleImpact, stateRoleRepresentative, stateRoleTrust, stateSolid, stateSpace, stateSpaceConfiguration, stateSpaceScale, stateSpectralBand, stateSpectralLine, stateStorm, stateSystem, stateThermodynamic, stateTime, stateTimeCycle, stateTimeFrequency, stateTimeGeologic, stateVisibility, sweet_v23Comments]

Your result should be a JSON dictionary with entities being the keys and categories being the values. There should be nothing in your answer except the JSON dictionary.

Here is an example:

entity list:
[
"depth first search",
"borealis",
"Magnolite",
"solar cycle",
]
your answer:
{
"depth first search": "reprMathSolution",
"borealis": "realmRegion",
"Magnolite": "matrMineral",
"solar cycle": "phenCycle",
}
"""
    entity_category_dict = chatbot.ask(prompt2)
    print(entity_category_dict)
    
    prompt3 = f"""
The following is the paragraph:

{paragraph}

The following is the entity list you have just generated

{entity_list}

Extract as many relations as possible from the paragraph. Your result should be a list of triples and nothing else. The first and third element in each triple should be in the entity list you have generated and the second element should be in the following relation category list. You should not extract any relation that is not in the following list. The relation you choose should be precise and diverse. You shouldn't use "RelatedTo" to describe all the relations.

[RelatedTo, FormOf, IsA, PartOf, HasA, UsedFor, CapableOf, AtLocation, Causes, HasSubevent, HasFirstSubevent, HasLastSubevent, HasPrerequisite, HasProperty, MotivatedByGoal, ObstructedBy, Desires, CreatedBy, Synonym, Antonym, DistinctFrom, DerivedFrom, SymbolOf, DefinedAs, MannerOf, LocatedNear, HasContext, SimilarTo, EtymologicallyRelatedTo, EtymologicallyDerivedFrom, CausesDesire, MadeOf, ReceivesAction, ExternalURL]

Here is an example.

paragraph: 
Scientists conducted depth first search into borealis and found that the formation of Magnolite is obstructed by solar cycle.

entity list:
[
    "depth first search",
    "borealis",
    "Magnolite",
    "solar cycle",
]

your answer:

[
    ["depth first search","UsedFor","borealis"],
    ["Magnolite","ObstructedBy","solar cycle"],
]

"""
    relation_list = chatbot.ask(prompt3)
    print(relation_list)
    
    try:
    
        p_entity_list = json.loads(entity_list)
        p_entity_category_dict = json.loads(entity_category_dict)
        p_relation_list = json.loads(relation_list)
        
        return [
            p_entity_list, 
            p_entity_category_dict,
            p_relation_list
            ]
    except:
        return (entity_list, entity_category_dict, relation_list)





""" 海洋组 @狄子钧 """
prompt = '''please describe the relationship between "{ Ontology[1] }" and "{ Ontology[2] }" using just one word, which is choosed from { Relations }.'''

from getchat import get_chat
from itertools import combinations
import pandas as pd
from pandas import ExcelWriter
import openpyxl

head = []
tail = []
relation = []

Entities = pd.read_excel("Physical_Conception.xlsx")
entity_comb = combinations(Entities.iloc[:, 0].tolist(), 2)
for entity in entity_comb:
    input_content = f'''please tell me the relationship between "{entity[0]}" and "{entity[1]}" using just one word, 
                    which is choosed from RelatedTo, FormOf, IsA, PartOf, HasA, UsedFor, CapableOf, AtLocation, 
                    Causes, HasSubevent, HasFirstSubevent, HasLastSubevent, HasPrerequisite, HasProperty, 
                    MotivatedByGoal, ObstructedBy, Desires, CreatedBy, Synonym, Antonym, DistinctFrom, DerivedFrom, 
                    SymbolOf, DefinedAs, MannerOf, LocatedNear, HasContext, SimilarTo, EtymologicallyRelatedTo, 
                    EtymologicallyDerivedFrom, CausesDesire, MadeOf, ReceivesAction, ExternalURL.'''
    response = get_chat(input_content)['choices'][0]['message']['content']
    print(response)
    head.append(entity[0])
    tail.append(entity[1])
    relation.append(response)

save_csv = pd.DataFrame({'Head entity': head, "Relation": relation, "Tail entity": tail})
with ExcelWriter("Result/part1/part_result.xlsx") as writer:
    save_csv.to_excel(writer, index=False, engine=openpyxl)



prompt = '''After reading the { Titles } these 10 papers, please describe the relationship between "{ Ontology[1] }" and "{ Ontology[2] }" using just one word, which is choosed from { Relations }.''' 

# ToDo：在数据库中检索每个关键词的top10文章，chat_gpt阅读文献后，“give me the relation of entity[0] and entity[1]”

import elasticsearch
import pandas as pd
from pandas import ExcelWriter
import openpyxl
from getchat import get_chat
from itertools import combinations
import numpy as np


def get_query(entities):
    index = "gakg_document"
    query_json = {"query": {
                        "bool": {
                            "must": [],
                            "filter": [
                                {
                                    "bool": {
                                        "filter": [
                                            {"multi_match": {
                                                    "type": "best_fields",
                                                    "query": "{}".format(entities[0]),
                                                }
                                            },
                                            {"multi_match": {
                                                    "type": "best_fields",
                                                    "query": "{}".format(entities[1]),
                                                }}]}}], }}}
    query = es.search(index=index, body=query_json)
    return query["hits"]["hits"]


# ToDo：提供给chat_gpt十篇文献标题后，查看head和tail的relation
prompt_1 = '''After reading {} these 10 papers, '''
prompt_2 = '''please describe the relationship between "{}" and "{}" using just one word, '''
prompt_3 = '''which is choosed from RelatedTo, FormOf, IsA, PartOf, HasA, UsedFor, CapableOf, AtLocation, 
                Causes, HasSubevent, HasFirstSubevent, HasLastSubevent, HasPrerequisite, HasProperty, 
                MotivatedByGoal, ObstructedBy, Desires, CreatedBy, Synonym, Antonym, DistinctFrom, DerivedFrom, 
                SymbolOf, DefinedAs, MannerOf, LocatedNear, HasContext, SimilarTo, EtymologicallyRelatedTo, 
                EtymologicallyDerivedFrom, CausesDesire, MadeOf, ReceivesAction, ExternalURL.'''

head = []
tail = []
relation = []
save_csv_path = "Result/part2/Database Extraction.xlsx"
writer1 = ExcelWriter(save_csv_path, engine='openpyxl')
save_relation_path = "Result/part2/part2_result.xlsx"
writer2 = ExcelWriter(save_relation_path, engine='openpyxl')

Entities = pd.read_excel("Physical_Conception.xlsx", header=None, usecols=[0])
entity_comb = combinations(Entities.iloc[:, 0].tolist(), 2)
for num, entity in enumerate(entity_comb):
    content = []
    title = []
    abstract = []
    url = []
    titles = ""
    for paper in get_query(entity):
        content.append(paper['_source']['content'])
        title.append(paper['_source']['name'])
        abstract.append(paper['_source']['abstract'])
        url.append(paper['_source']['url'])
        titles += paper['_source']['name']

    save_csv = pd.DataFrame({'Title': title, "Abstract": abstract, "Content": content, "url": url})
    insert_column = [", ".join(entity), "", "", ""]
    save_csv = pd.DataFrame(np.insert(save_csv.values, 0, values=insert_column, axis=0))
    save_csv.to_excel(writer1,
                      sheet_name="sheet{}".format(num),
                      index=False,
                      engine=openpyxl)

    input_content = prompt_1.format(titles) + prompt_2.format(entity[0], entity[1]) + prompt_3
    response = get_chat(input_content)['choices'][0]['message']['content']
    print(response)
    head.append(entity[0])
    tail.append(entity[1])
    relation.append(response)

    save_relation = pd.DataFrame({'Head entity': head, "Relation": relation, "Tail entity": tail})
    save_relation.to_excel(writer2,
                           index=False,
                           engine=openpyxl)

writer1.save()
writer2.save()