from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


_TARGETS = {
    "deepseek": "KUN Personal Knowledge Agent/DeepSeek API Key",
    "dashscope": "KUN Personal Knowledge Agent/DashScope API Key",
}

CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def _target(provider: str) -> str:
    try:
        return _TARGETS[provider]
    except KeyError as error:
        raise ValueError(f"未知凭据提供方：{provider}") from error


def available() -> bool:
    return sys.platform == "win32"


def set_secret(provider: str, secret: str) -> None:
    value = secret.strip()
    if not value:
        raise ValueError("凭据不能为空")
    if not available():
        raise RuntimeError("Windows 凭据管理器仅在 Windows 上可用")

    encoded = value.encode("utf-8")
    blob = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
    credential = CREDENTIALW()
    credential.Type = CRED_TYPE_GENERIC
    credential.TargetName = _target(provider)
    credential.CredentialBlobSize = len(encoded)
    credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    credential.Persist = CRED_PERSIST_LOCAL_MACHINE
    credential.UserName = f"kun-{provider}"

    cred_write = ctypes.windll.advapi32.CredWriteW
    cred_write.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
    cred_write.restype = wintypes.BOOL
    if not cred_write(ctypes.byref(credential), 0):
        raise ctypes.WinError()


def get_secret(provider: str) -> str:
    if not available():
        return ""
    pointer = ctypes.POINTER(CREDENTIALW)()
    cred_read = ctypes.windll.advapi32.CredReadW
    cred_read.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(CREDENTIALW)),
    ]
    cred_read.restype = wintypes.BOOL
    if not cred_read(_target(provider), CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        return ""
    try:
        credential = pointer.contents
        if not credential.CredentialBlob or not credential.CredentialBlobSize:
            return ""
        raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
        return raw.decode("utf-8")
    finally:
        ctypes.windll.advapi32.CredFree(pointer)


def delete_secret(provider: str) -> None:
    if not available():
        return
    delete = ctypes.windll.advapi32.CredDeleteW
    delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    delete.restype = wintypes.BOOL
    delete(_target(provider), CRED_TYPE_GENERIC, 0)
