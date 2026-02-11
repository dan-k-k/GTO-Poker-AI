# app/_visuals.py

import os
from PIL import Image, ImageDraw, ImageFont
from app.poker_core import card_to_string, GameState

# This class acts as a central "control panel" for all visual settings.
class LayoutManager:
    def __init__(self):
        
        # Overall Image Size
        self.IMG_WIDTH = 600
        self.IMG_HEIGHT = 400

        # Table Appearance
        self.TABLE_BORDER_WIDTH = 50 # How thick the 'wood' border is
        self.FELT_RADIUS = 40

        # Card Sizing (I've made them slightly smaller)
        self.CARD_SCALE = 0.6 # Master scale for cards. 1.0 is default, 0.9 is 90%.
        self.CARD_WIDTH = int(90 * self.CARD_SCALE)
        self.CARD_HEIGHT = int(130 * self.CARD_SCALE)
        
        # Font Sizing
        self.CARD_RANK_FONT_SIZE = int(36 * self.CARD_SCALE)
        self.CARD_SUIT_FONT_SIZE = int(28 * self.CARD_SCALE)
        self.UI_FONT_SIZE = 18
        
        # Spacing
        self.PLAYER_VERTICAL_OFFSET = -10 # How far player cards are from the top/bottom edge
        self.COMMUNITY_CARD_SPACING = 10
        self.DEALER_BUTTON_OFFSET = 30

        # --- Calculated Positions ---
        self.center_x, self.center_y = self.IMG_WIDTH / 2, self.IMG_HEIGHT / 2
        
        # Player 0 (Bottom)
        self.p0_y = self.IMG_HEIGHT - self.TABLE_BORDER_WIDTH - self.PLAYER_VERTICAL_OFFSET
        self.p0_card_y = self.p0_y - self.CARD_HEIGHT / 2
        self.p0_card1_x = self.center_x - self.CARD_WIDTH - 5
        self.p0_card2_x = self.center_x + 5
        self.p0_stack_pos = (self.center_x, self.p0_y + self.CARD_HEIGHT / 2 + 20)
        self.p0_bet_pos = (self.center_x, self.p0_y - self.CARD_HEIGHT / 2 - 25)
        self.p0_dealer_pos = (self.p0_card1_x - self.DEALER_BUTTON_OFFSET, self.p0_y)

        # Player 1 (Top)
        self.p1_y = self.TABLE_BORDER_WIDTH + self.PLAYER_VERTICAL_OFFSET
        self.p1_card_y = self.p1_y - self.CARD_HEIGHT / 2
        self.p1_card1_x = self.center_x - self.CARD_WIDTH - 5
        self.p1_card2_x = self.center_x + 5
        self.p1_stack_pos = (self.center_x, self.p1_y + self.CARD_HEIGHT / 2 + 20)
        self.p1_bet_pos = (self.center_x, self.p1_y - self.CARD_HEIGHT / 2 - 25)
        self.p1_dealer_pos = (self.p1_card1_x - self.DEALER_BUTTON_OFFSET, self.p1_y)

        # Community Cards & Pot
        self.community_y = self.center_y - self.CARD_HEIGHT / 2
        self.pot_pos = (self.center_x, self.center_y + self.CARD_HEIGHT / 2 + 15)

# --- FONT & COLOR SETUP ---
FONT_PATH = os.path.join("static", "fonts", "DejaVuSans.ttf")
TABLE_COLOR = (34, 34, 34); FELT_COLOR = (42, 98, 61); BORDER_COLOR = (68, 68, 68)
WHITE = (255, 255, 255); BLACK = (0, 0, 0); RED = (217, 83, 79)
BUTTON_COLOR = (220, 220, 220); BUTTON_TEXT_COLOR = (0, 0, 0)
SUITS = {'c': '♣', 'd': '♦', 'h': '♥', 's': '♠'}
SUIT_COLORS = {'c': BLACK, 'd': RED, 'h': RED, 's': BLACK}

# --- Drawing Functions ---
def _load_fonts(layout: LayoutManager):
    try:
        ui_font = ImageFont.truetype(FONT_PATH, layout.UI_FONT_SIZE)
        card_rank_font = ImageFont.truetype(FONT_PATH, layout.CARD_RANK_FONT_SIZE)
        card_suit_font = ImageFont.truetype(FONT_PATH, layout.CARD_SUIT_FONT_SIZE)
        return ui_font, card_rank_font, card_suit_font
    except IOError:
        return ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()

def _draw_card(draw, pos, card_str, layout, fonts, face_down=False):
    x, y = pos
    _, rank_font, suit_font = fonts
    draw.rounded_rectangle([x, y, x + layout.CARD_WIDTH, y + layout.CARD_HEIGHT], radius=8, fill=WHITE, outline="gray", width=2)
    if face_down:
        draw.line([x+10, y+10, x+layout.CARD_WIDTH-10, y+layout.CARD_HEIGHT-10], fill=RED, width=3)
        draw.line([x+layout.CARD_WIDTH-10, y+10, x+10, y+layout.CARD_HEIGHT-10], fill=RED, width=3)
        return

    rank, suit_char = card_str[:-1], card_str[-1]
    suit_symbol = SUITS.get(suit_char, '?'); color = SUIT_COLORS.get(suit_char, BLACK)
    rank_pos = (x + layout.CARD_WIDTH / 2, y + layout.CARD_HEIGHT / 2 - (10 * layout.CARD_SCALE))
    suit_pos = (x + layout.CARD_WIDTH / 2, y + layout.CARD_HEIGHT / 2 + (20 * layout.CARD_SCALE))
    draw.text(rank_pos, rank, font=rank_font, fill=color, anchor="mm")
    draw.text(suit_pos, suit_symbol, font=suit_font, fill=color, anchor="mm")

def create_table_image(state: GameState, env, show_all_cards=False) -> Image:
    layout = LayoutManager()
    fonts = _load_fonts(layout)
    ui_font, _, _ = fonts
    
    img = Image.new('RGB', (layout.IMG_WIDTH, layout.IMG_HEIGHT), TABLE_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw table structure
    draw.rounded_rectangle([10, 10, layout.IMG_WIDTH - 10, layout.IMG_HEIGHT - 10], radius=50, fill=BORDER_COLOR)
    draw.rounded_rectangle(
        [layout.TABLE_BORDER_WIDTH, layout.TABLE_BORDER_WIDTH, layout.IMG_WIDTH - layout.TABLE_BORDER_WIDTH, layout.IMG_HEIGHT - layout.TABLE_BORDER_WIDTH], 
        radius=layout.FELT_RADIUS, fill=FELT_COLOR
    )

    # Player positions mapping
    player_info = {
        0: {'card1_pos': (layout.p0_card1_x, layout.p0_card_y), 'card2_pos': (layout.p0_card2_x, layout.p0_card_y), 'stack_pos': layout.p0_stack_pos, 'bet_pos': layout.p0_bet_pos, 'dealer_pos': layout.p0_dealer_pos},
        1: {'card1_pos': (layout.p1_card1_x, layout.p1_card_y), 'card2_pos': (layout.p1_card2_x, layout.p1_card_y), 'stack_pos': layout.p1_stack_pos, 'bet_pos': layout.p1_bet_pos, 'dealer_pos': layout.p1_dealer_pos},
    }

    # Draw players
    for i in range(state.num_players):
        info = player_info[i]
        show_cards = (i == 0) or show_all_cards
        
        if show_cards and state.hole_cards and len(state.hole_cards) > i:
            _draw_card(draw, info['card1_pos'], card_to_string(state.hole_cards[i][0]), layout, fonts)
            _draw_card(draw, info['card2_pos'], card_to_string(state.hole_cards[i][1]), layout, fonts)
        else:
            _draw_card(draw, info['card1_pos'], None, layout, fonts, face_down=True)
            _draw_card(draw, info['card2_pos'], None, layout, fonts, face_down=True)

        draw.text(info['stack_pos'], f"Stack: {state.stacks[i]}", font=ui_font, fill=WHITE, anchor="mm")
        if state.current_bets[i] > 0:
            draw.text(info['bet_pos'], f"Bet: {state.current_bets[i]}", font=ui_font, fill=(240, 200, 100), anchor="mm")
        
        if state.dealer_pos == i:
            dx, dy = info['dealer_pos']
            draw.ellipse([dx-15, dy-15, dx+15, dy+15], fill=BUTTON_COLOR)
            draw.text((dx, dy), "D", font=ui_font, fill=BUTTON_TEXT_COLOR, anchor="mm")

    # Draw community cards
    num_comm = len(state.community)
    total_comm_width = (num_comm * layout.CARD_WIDTH) + max(0, num_comm - 1) * layout.COMMUNITY_CARD_SPACING
    start_x = (layout.IMG_WIDTH - total_comm_width) / 2
    for i, card_id in enumerate(state.community):
        card_x = start_x + i * (layout.CARD_WIDTH + layout.COMMUNITY_CARD_SPACING)
        _draw_card(draw, (card_x, layout.community_y), card_to_string(card_id), layout, fonts)

    # Draw pot
    draw.text(layout.pot_pos, f"Pot: {state.pot}", font=ui_font, fill=WHITE, anchor="mm")
    
    return img

