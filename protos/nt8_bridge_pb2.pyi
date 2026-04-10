from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CandleRequest(_message.Message):
    __slots__ = ("symbol", "open", "high", "low", "close", "volume", "time", "current_position", "current_pnl")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    OPEN_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    LOW_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    CURRENT_POSITION_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PNL_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    time: str
    current_position: float
    current_pnl: float
    def __init__(self, symbol: _Optional[str] = ..., open: _Optional[float] = ..., high: _Optional[float] = ..., low: _Optional[float] = ..., close: _Optional[float] = ..., volume: _Optional[float] = ..., time: _Optional[str] = ..., current_position: _Optional[float] = ..., current_pnl: _Optional[float] = ...) -> None: ...

class ActionResponse(_message.Message):
    __slots__ = ("action", "confidence", "message")
    class Action(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        HOLD: _ClassVar[ActionResponse.Action]
        BUY: _ClassVar[ActionResponse.Action]
        SELL: _ClassVar[ActionResponse.Action]
        CLOSE_ALL: _ClassVar[ActionResponse.Action]
    HOLD: ActionResponse.Action
    BUY: ActionResponse.Action
    SELL: ActionResponse.Action
    CLOSE_ALL: ActionResponse.Action
    ACTION_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    action: ActionResponse.Action
    confidence: float
    message: str
    def __init__(self, action: _Optional[_Union[ActionResponse.Action, str]] = ..., confidence: _Optional[float] = ..., message: _Optional[str] = ...) -> None: ...
