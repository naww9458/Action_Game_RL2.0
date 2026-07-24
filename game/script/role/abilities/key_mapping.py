
class KeyMapping:
    Keyboard_Mappings = None
    Mouse_Mappings = {
        "left": 0,
        "middle": 1,
        "right": 3, # I'm not sure why it is mapping to 3 in viewerGL, but "It just work"
    }

    @classmethod
    def get(cls, keys: dict | list | str, default=None):
        """
        安全獲取鍵位的方法
        例如: KeyMapping.get("a") -> 返回 pygame.K_a
        """
        if isinstance(keys, list):
            for i in range(len(keys)):
                keys[i] = cls.Keyboard_Mappings.get(keys[i].lower(), default)
        
        elif isinstance(keys, dict):
            for key, key_list in keys["keyboard"].items():
                keys["keyboard"][key] = [cls.Keyboard_Mappings.get(key_str.lower(), default) for key_str in key_list]

            for key, key_list in keys["mouse"].items():
                keys["mouse"][key] = [cls.Mouse_Mappings.get(key_str.lower(), default) for key_str in key_list]
                
        elif isinstance(keys, str):
            keys = cls.Keyboard_Mappings.get(keys.lower(), default)

        else:
            raise TypeError("KeyMapping.get only accepts str, list, or dict types.")
        
        return keys
