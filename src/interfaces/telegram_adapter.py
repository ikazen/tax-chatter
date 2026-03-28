"""python-telegram-bot 기반 ChatAdapter 구현체."""
import logging

from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler as TGMessageHandler,
    filters,
)

from config.settings import settings
from src.interfaces.base import ChatAdapter, MessageHandler

logger = logging.getLogger(__name__)


class TelegramAdapter(ChatAdapter):
    """Telegram 봇 인터페이스."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token or settings.telegram_bot_token
        self._handler: MessageHandler | None = None
        self._app: Application | None = None  # type: ignore[type-arg]

    def register_handler(self, handler: MessageHandler) -> None:
        """메시지 수신 시 호출될 핸들러를 등록한다."""
        self._handler = handler

    async def send_message(self, user_id: str, text: str) -> None:
        """사용자에게 메시지를 전송한다."""
        if self._app is None:
            raise RuntimeError("봇이 시작되지 않았습니다")
        await self._app.bot.send_message(chat_id=int(user_id), text=text)

    async def start(self) -> None:
        """Telegram 봇을 시작한다."""
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(
            TGMessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling()
        logger.info("Telegram 봇 시작됨")

    async def stop(self) -> None:
        """Telegram 봇을 종료한다."""
        if self._app is None:
            return
        if self._app.updater:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram 봇 종료됨")

    async def _on_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Telegram 메시지 수신 콜백."""
        if self._handler is None or update.message is None or update.message.text is None:
            return
        user_id = str(update.effective_user.id) if update.effective_user else "unknown"
        text = update.message.text
        response = await self._handler(user_id, text)
        await update.message.reply_text(response)
