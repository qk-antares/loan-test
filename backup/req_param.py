from typing import List, Optional
from decimal import Decimal

class IdInfo:
    idNumber: str
    identityType: str
    name: str
    gender: str
    nation: str
    address: str
    issuedBy: str
    validityDate: str
    birthDate: str

    def __init__(self, idNumber: str, identityType: str, name: str, gender: str, nation: str, address: str, issuedBy: str, validityDate: str, birthDate: str):
        self.idNumber = idNumber
        self.identityType = identityType
        self.name = name
        self.gender = gender
        self.nation = nation
        self.address = address
        self.issuedBy = issuedBy
        self.validityDate = validityDate
        self.birthDate = birthDate

class PictureInfo:
    pictureType: str
    pictureContent: str
    photoSuffix: str
    faceChannel: Optional[str]
    faceScore: Optional[str]
    collectTime: Optional[str]

    def __init__(self, pictureType: str, pictureContent: str, photoSuffix: str, faceChannel: Optional[str]=None, faceScore: Optional[str]=None, collectTime: Optional[str]=None):
        self.pictureType = pictureType
        self.pictureContent = pictureContent
        self.photoSuffix = photoSuffix
        self.faceChannel = faceChannel
        self.faceScore = faceScore
        self.collectTime = collectTime

class Linkman:
    name: str
    phone: str
    relationship: str

    def __init__(self, name: str, phone: str, relationship: str):
        self.name = name
        self.phone = phone
        self.relationship = relationship

class CompanyInfo:
    companyName: str
    companyAddress: str
    occupation: str
    industry: str

    def __init__(self, companyName: str, companyAddress: str, occupation: str, industry: str):
        self.companyName = companyName
        self.companyAddress = companyAddress
        self.occupation = occupation
        self.industry = industry

class DeviceInfo:
    osType: str
    gpsLongitude: str
    gpsLatitude: str
    deviceId: Optional[str]
    isCrossDomain: Optional[bool]
    macId: Optional[str]
    phoneType: Optional[str]
    phoneMaker: Optional[str]
    ipAddress: Optional[str]
    memory: Optional[int]
    storage: Optional[int]
    unStorage: Optional[int]
    electricity: Optional[float]
    dns: Optional[str]
    deviceCode: Optional[str]
    sysType: Optional[str]
    operateCode: Optional[str]
    androidId: Optional[str]
    identificationCode: Optional[str]
    applyPos: str

    def __init__(self, osType: str, gpsLongitude: str, gpsLatitude: str, applyPos: str, deviceId: Optional[str]=None, isCrossDomain: Optional[bool]=None, macId: Optional[str]=None, phoneType: Optional[str]=None, phoneMaker: Optional[str]=None, ipAddress: Optional[str]=None, memory: Optional[int]=None, storage: Optional[int]=None, unStorage: Optional[int]=None, electricity: Optional[float]=None, dns: Optional[str]=None, deviceCode: Optional[str]=None, sysType: Optional[str]=None, operateCode: Optional[str]=None, androidId: Optional[str]=None, identificationCode: Optional[str]=None):
        self.osType = osType
        self.gpsLongitude = gpsLongitude
        self.gpsLatitude = gpsLatitude
        self.deviceId = deviceId
        self.isCrossDomain = isCrossDomain
        self.macId = macId
        self.phoneType = phoneType
        self.phoneMaker = phoneMaker
        self.ipAddress = ipAddress
        self.memory = memory
        self.storage = storage
        self.unStorage = unStorage
        self.electricity = electricity
        self.dns = dns
        self.deviceCode = deviceCode
        self.sysType = sysType
        self.operateCode = operateCode
        self.androidId = androidId
        self.identificationCode = identificationCode
        self.applyPos = applyPos

class BankCardInfo:
    cardType: Optional[str]
    bankCode: Optional[str]
    bankCardNo: Optional[str]
    reservePhoneNo: Optional[str]
    bankName: Optional[str]

    def __init__(self, cardType: Optional[str]=None, bankCode: Optional[str]=None, bankCardNo: Optional[str]=None, reservePhoneNo: Optional[str]=None, bankName: Optional[str]=None):
        self.cardType = cardType
        self.bankCode = bankCode
        self.bankCardNo = bankCardNo
        self.reservePhoneNo = reservePhoneNo
        self.bankName = bankName

class LoanRequest:
    userId: str
    orderId: str
    phone: str
    degree: str
    maritalStatus: str
    province: str
    city: str
    area: str
    liveAddress: str
    income: str
    amount: Optional[Decimal]
    term: Optional[int]
    email: Optional[str]
    jobFunctions: Optional[str]
    resideFunctions: Optional[str]
    purpose: str
    customerSource: str
    idInfo: IdInfo
    linkmanList: List[Linkman]
    companyInfo: CompanyInfo
    deviceInfo: DeviceInfo
    pictureInfo: List[PictureInfo]
    bankCardInfo: Optional[BankCardInfo]

    def __init__(self, userId: str, orderId: str, phone: str, degree: str, maritalStatus: str, province: str, city: str, area: str, liveAddress: str, income: str, amount: Optional[Decimal], term: Optional[int], email: Optional[str], jobFunctions: Optional[str], resideFunctions: Optional[str], purpose: str, customerSource: str, idInfo: IdInfo, linkmanList: List[Linkman], companyInfo: CompanyInfo, deviceInfo: DeviceInfo, pictureInfo: List[PictureInfo], bankCardInfo: Optional[BankCardInfo]=None):
        self.userId = userId
        self.orderId = orderId
        self.phone = phone
        self.degree = degree
        self.maritalStatus = maritalStatus
        self.province = province
        self.city = city
        self.area = area
        self.liveAddress = liveAddress
        self.income = income
        self.amount = amount
        self.term = term
        self.email = email
        self.jobFunctions = jobFunctions
        self.resideFunctions = resideFunctions
        self.purpose = purpose
        self.customerSource = customerSource
        self.idInfo = idInfo
        self.linkmanList = linkmanList
        self.companyInfo = companyInfo
        self.deviceInfo = deviceInfo
        self.pictureInfo = pictureInfo
        self.bankCardInfo = bankCardInfo